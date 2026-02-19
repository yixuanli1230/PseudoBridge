import json
import os
import time
import logging
from typing import List, Dict, Optional, Tuple, Any

import config
from evaluator import PseudoCodeEvaluator
from refiner import PseudoCodeRefiner

# Import vLLM only if needed or safeguard import
try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

class PseudoCodeGenerator:
    def __init__(self):
        # 1. Setup OpenAI Client (Used for Evaluator/Refiner, and Generator if mode is API)
        from openai import OpenAI
        import tiktoken
        self.client = OpenAI(api_key=config.OPENAI_API_KEY, base_url=config.BASE_URL)
        
        # 2. Initialize Evaluator and Refiner
        self.evaluator = PseudoCodeEvaluator(self.client)
        self.refiner = PseudoCodeRefiner(self.client)
        
        # 3. Initialize Tokenizer (for counting only)
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except:
            self.encoding = None

        # 4. Setup Generator (Local vLLM or API)
        self.mode = config.GENERATION_MODE
        self.llm = None
        self.sampling_params = None

        if self.mode == "local":
            if LLM is None:
                raise ImportError("vLLM is not installed. Please install it or set GENERATION_MODE='api'.")
            
            logger.info(f"🚀 Loading Local Model: {config.LOCAL_MODEL_PATH} ...")
            self.llm = LLM(
                model=config.LOCAL_MODEL_PATH,
                gpu_memory_utilization=config.GPU_MEMORY_UTILIZATION,
                max_model_len=config.MAX_MODEL_LEN,
                trust_remote_code=True
            )
            self.sampling_params = SamplingParams(
                temperature=0.6,
                top_p=0.95,
                max_tokens=2048,
                stop=["<|endoftext|>", "<|im_end|>"]
            )
            logger.info("✅ Local Model Loaded Successfully.")

        self.SYSTEM_PROMPT = """
You are a professional Python developer and now need to convert the given docstring content into standardised pseudo-code. Please strictly follow the following specification requirements:

1. Comments: use `//` for single line comments s, but do not use multi-line comments.
2. Variables: Choose clear typed names for variables.
3. Input/Output: Keep input/output simple and clear.
4. Conditional Clauses: Use `IF`, `ELSIF` and `ELSE` with proper indentation.
5. Loops: Use `FOR`, `WHILE` or `DO.... .WHILE` loops, specify conditions and indent the code.
6. Functions/Procedures: name functions descriptively and consider arguments. 
7. Formatting: Maintain consistent 2-4 space indentation for clarity.
8. Content: Give only the key steps and ignore details.

Based on the contents of the provided docstring, please write the appropriate pseudo-code for the solution according to the pseudo-code standard. Output only the generated solution pseudo-code and do not include the docstring content in the output.
"""

    def _count_tokens(self, text: str) -> int:
        """Helper to count tokens for a given string."""
        if not text or not self.encoding:
            return 0
        return len(self.encoding.encode(text))

    def _prepare_prompt(self, docstring: str) -> str:
        """Combine system prompt and user content based on model type."""
        # For Chat Models (Instruct), we usually format it nicely. 
        # For simplicity, we append strings. Ideally, apply_chat_template should be used for local models.
        user_content = f"Generate pseudo code for:\n{docstring}"
        
        if self.mode == "local":
            # Simple formatting for vLLM raw input. 
            # Note: If your local model requires strict ChatML (<|im_start|...>), adjust here.
            # This is a generic prompt format.
            return f"{self.SYSTEM_PROMPT}\n\nUser: {user_content}\n\nAssistant:"
        else:
            return user_content

    def generate_batch_initial(self, docstrings: List[str]) -> Tuple[List[str], List[dict]]:
        """
        Generate initial pseudo-code for a batch of docstrings.
        Returns: (List of codes, List of usage stats)
        """
        results_content = []
        usage_stats = []

        if self.mode == "local":
            # --- Local vLLM Generation ---
            prompts = [self._prepare_prompt(d) for d in docstrings]
            
            # vLLM handles batching internally efficiently
            outputs = self.llm.generate(prompts, self.sampling_params)

            for output in outputs:
                generated_text = output.outputs[0].text.strip()
                results_content.append(generated_text)
                
                # Estimate usage for local
                prompt_toks = len(output.prompt_token_ids)
                output_toks = len(output.outputs[0].token_ids)
                usage_stats.append({
                    "input_tokens": prompt_toks,
                    "output_tokens": output_toks
                })

        else:
            # --- API Generation (Sequential Loop for simplicity) ---
            # To make this truly parallel for API, one would use asyncio. 
            # Keeping it simple loop here to match previous logic logic structure.
            for docstring in docstrings:
                code, in_tok, out_tok = self._generate_single_api(docstring)
                results_content.append(code)
                usage_stats.append({
                    "input_tokens": in_tok,
                    "output_tokens": out_tok
                })
        
        return results_content, usage_stats

    def _generate_single_api(self, docstring: str) -> Tuple[str, int, int]:
        """Internal helper for single API call."""
        user_content = self._prepare_prompt(docstring)
        for attempt in range(config.MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=config.GENERATOR_MODEL,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    stream=False,
                    temperature=0.6,
                    top_p=0.95
                )
                content = response.choices[0].message.content.strip()
                
                if hasattr(response, 'usage') and response.usage:
                    return content, response.usage.prompt_tokens, response.usage.completion_tokens
                else:
                    return content, 0, 0
            except Exception as e:
                logger.warning(f"API Request Failed (Attempt {attempt+1}): {e}")
                time.sleep(1)
        return "", 0, 0

    def evaluate_and_refine(self, data_item: Dict, initial_code: str, usage_stat: Dict) -> Optional[Tuple[Dict, Dict]]:
        """
        Run the Evaluator -> Refiner loop for a single item.
        """
        line_num = data_item.get("line_id", "unknown")
        docstring = data_item["docstring"]
        original_code = data_item["code"]
        pseudo_code = initial_code

        current_input_tokens = usage_stat["input_tokens"]
        current_output_tokens = usage_stat["output_tokens"]

        if not pseudo_code:
            return None

        final_data = None
        max_loops = 3

        print(f"\n🔹 [Initial Generation - Row {line_num}]:\n{pseudo_code}...")

        for attempt in range(max_loops + 1):
            iter_info = f"Iter {attempt}"
            
            # --- Stage 1: Alignment Check ---
            # Estimate tokens
            eval_input_est = 500 + self._count_tokens(docstring) + self._count_tokens(original_code) + self._count_tokens(pseudo_code)
            current_input_tokens += eval_input_est
            
            align_res = self.evaluator.verify_alignment(docstring, original_code, pseudo_code)
            current_output_tokens += self._count_tokens(json.dumps(align_res))

            if not align_res.get("is_aligned", False):
                issues = align_res.get("issues", "Logic mismatch")
                print(f"   ⚠️ Alignment Failed: {issues}")
                
                if attempt < max_loops:
                    # Refinement
                    refine_input_est = 400 + self._count_tokens(docstring) + self._count_tokens(original_code) + self._count_tokens(pseudo_code) + self._count_tokens(issues)
                    current_input_tokens += refine_input_est
                    
                    pseudo_code = self.refiner.refine(
                        docstring, original_code, pseudo_code, 
                        feedback=issues, 
                        issue_type="alignment"
                    )
                    current_output_tokens += self._count_tokens(pseudo_code)
                    continue 
                else:
                    logger.warning(f"Row {line_num}: ❌ Failed Alignment. Discarding.")
                    return None 

            # --- Stage 2: Quality Scoring ---
            current_input_tokens += eval_input_est 
            quality_res = self.evaluator.evaluate_quality(docstring, original_code, pseudo_code)
            scores = quality_res.get("scores", {})
            current_output_tokens += self._count_tokens(json.dumps(quality_res))
            
            is_high_quality = quality_res.get("all_passed", False)

            if is_high_quality:
                final_data = {
                    "pseudo_code": pseudo_code,
                    "quality_scores": scores,
                    "refinement_rounds": attempt
                }
                logger.info(f"Row {line_num}: ✅ Accepted (Scores: {scores})")
                break
            
            # Quality failed, refine
            feedback = quality_res.get("feedback", "Improve quality scores")
            print(f"   ⚠️ Quality Feedback: {feedback}")
            
            if attempt < max_loops:
                refine_input_est = 400 + self._count_tokens(docstring) + self._count_tokens(original_code) + self._count_tokens(pseudo_code) + self._count_tokens(feedback)
                current_input_tokens += refine_input_est
                
                pseudo_code = self.refiner.refine(
                    docstring, original_code, pseudo_code, 
                    feedback=feedback, 
                    issue_type="quality"
                )
                current_output_tokens += self._count_tokens(pseudo_code)
            else:
                logger.warning(f"Row {line_num}: ❌ Low Quality after retries. Discarding.")
                return None

        if final_data:
            result = data_item.copy()
            result.update(final_data)
            token_info = {
                "line_id": line_num,
                "input_token": current_input_tokens,
                "output_token": current_output_tokens
            }
            return result, token_info
        
        return None

    def run_batch(self):
        """Main batch processing flow with support for batched generation."""
        temp_files = []
        batch_data_buffer = [] # Holds raw lines to process
        
        processed_results = []
        token_stats_list = []
        
        file_counter = 1
        total_accepted = 0
        
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        token_log_file = os.path.join(config.OUTPUT_DIR, "token_num.jsonl")
        
        logger.info(f"Starting Processing via Mode: [{self.mode}]")
        if self.mode == "local":
            logger.info(f"Model: {config.GENERATOR_MODEL} | Batch Size: {config.VLLM_BATCH_SIZE}")
        
        try:
            with open(config.INPUT_FILE, "r", encoding="utf-8") as f:
                
                # Read loop
                for i, line in enumerate(f, start=1):
                    if i < config.START_LINE: continue
                    if config.END_LINE and i > config.END_LINE: break
                    
                    try:
                        data = json.loads(line.strip())
                        data['line_id'] = i
                        if "docstring" in data and "code" in data:
                            batch_data_buffer.append(data)
                    except json.JSONDecodeError:
                        continue

                    # Process when buffer reaches batch size
                    target_batch_size = config.VLLM_BATCH_SIZE if self.mode == "local" else 1 # API usually 1 or small
                    
                    if len(batch_data_buffer) >= target_batch_size:
                        self._process_data_chunk(batch_data_buffer, processed_results, token_stats_list)
                        batch_data_buffer = []

                        # Save checkpoint
                        if len(processed_results) >= config.SAVE_INTERVAL:
                            self._save_temp(processed_results, file_counter)
                            self._save_token_stats(token_stats_list, token_log_file)
                            temp_files.append(f"processed_{file_counter}.jsonl")
                            processed_results = []
                            token_stats_list = []
                            file_counter += 1
                
                # Process remaining
                if batch_data_buffer:
                    self._process_data_chunk(batch_data_buffer, processed_results, token_stats_list)

                # Save remaining
                if processed_results:
                    self._save_temp(processed_results, file_counter)
                    self._save_token_stats(token_stats_list, token_log_file)
                    temp_files.append(f"processed_{file_counter}.jsonl")

            # Merge
            if temp_files:
                self._merge_files(temp_files)
                logger.info("🎉 Processing Complete.")
            
        except FileNotFoundError:
            logger.error(f"Input file not found: {config.INPUT_FILE}")

    def _process_data_chunk(self, data_chunk: List[Dict], results_list: List, tokens_list: List):
        """Helper to process a chunk of data (Generate -> Eval/Refine)"""
        if not data_chunk: return

        # 1. Batch Generation (Parallel if local)
        docstrings = [d["docstring"] for d in data_chunk]
        initial_codes, usage_stats = self.generate_batch_initial(docstrings)

        # 2. Sequential Evaluation/Refinement (Iterate through results)
        for idx, item in enumerate(data_chunk):
            res_tuple = self.evaluate_and_refine(item, initial_codes[idx], usage_stats[idx])
            if res_tuple:
                res_data, res_token = res_tuple
                results_list.append(res_data)
                tokens_list.append(res_token)

    def _save_temp(self, data: List[Dict], index: int):
        path = os.path.join(config.OUTPUT_DIR, f"processed_{index}.jsonl")
        try:
            with open(path, "w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            logger.info(f"Saved temp file: {path} ({len(data)} records)")
        except IOError as e:
            logger.error(f"Failed to save temp file: {e}")
            
    def _save_token_stats(self, token_data: List[Dict], filepath: str):
        if not token_data: return
        try:
            with open(filepath, "a", encoding="utf-8") as f:
                for item in token_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        except IOError as e:
            logger.error(f"Failed to save token stats: {e}")

    def _merge_files(self, filenames: List[str]):
        logger.info(f"Merging files into {config.FINAL_OUTPUT_FILE}...")
        try:
            with open(config.FINAL_OUTPUT_FILE, "w", encoding="utf-8") as outfile:
                for fname in filenames:
                    path = os.path.join(config.OUTPUT_DIR, fname)
                    if os.path.exists(path):
                        with open(path, "r", encoding="utf-8") as infile:
                            outfile.write(infile.read())
            logger.info(f"✅ Merged file saved.")
        except IOError as e:
            logger.error(f"Failed to merge files: {e}")

if __name__ == "__main__":
    app = PseudoCodeGenerator()
    app.run_batch()