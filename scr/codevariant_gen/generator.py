import json
import os
import re
import time
import logging
from typing import List, Dict, Optional, Tuple

import config
from style_manager import StyleManager
from evaluator import VariantEvaluator
from refiner import VariantRefiner

# Import vLLM if configured for local use anywhere in the pipeline
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

class CodeVariantGenerator:
    def __init__(self):
        from openai import OpenAI
        import tiktoken
        
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except:
            self.encoding = None

        self.style_manager = StyleManager()
        
        # Dictionary to store local vLLM instances and share them if paths match 
        # (Saves massive GPU memory if Gen/Eval/Refine use the same local model)
        self.local_llms = {}

        def get_local_llm(model_path, gpu_mem, max_len):
            if LLM is None:
                raise ImportError("vLLM is not installed. Please set modes to 'api' or install vllm.")
            if model_path not in self.local_llms:
                logger.info(f"🚀 Loading Local Model: {model_path} (GPU Mem: {gpu_mem})")
                self.local_llms[model_path] = LLM(
                    model=model_path,
                    tensor_parallel_size=1,
                    trust_remote_code=True,
                    max_model_len=max_len,
                    enforce_eager=True,
                    gpu_memory_utilization=gpu_mem
                )
            return self.local_llms[model_path]

        # 1. Initialize Generator Engine
        self.gen_mode = config.GEN_MODE
        if self.gen_mode == "local":
            self.gen_llm = get_local_llm(config.GEN_MODEL, config.GEN_GPU_MEM, config.GEN_MAX_MODEL_LEN)
            self.gen_client = None
            self.gen_sampling_params = SamplingParams(
                temperature=0.6,
                top_p=0.9,
                max_tokens=config.GEN_MAX_TOKENS,
                stop=["<|endoftext|>", "<|im_end|>", "</VERSION>"]
            )
            logger.info("✅ Generator Local Model Loaded.")
        else:
            logger.info("☁️ Generator operating in API Mode.")
            self.gen_llm = None
            self.gen_client = OpenAI(api_key=config.GEN_API_KEY, base_url=config.GEN_BASE_URL)

        # 2. Initialize Evaluator Engine
        eval_client, eval_llm = None, None
        if config.EVAL_MODE == "local":
            eval_llm = get_local_llm(config.EVAL_MODEL, config.EVAL_GPU_MEM, config.EVAL_MAX_MODEL_LEN)
        else:
            eval_client = OpenAI(api_key=config.EVAL_API_KEY, base_url=config.EVAL_BASE_URL)
        self.evaluator = VariantEvaluator(mode=config.EVAL_MODE, client=eval_client, llm=eval_llm)

        # 3. Initialize Refiner Engine
        ref_client, ref_llm = None, None
        if config.REF_MODE == "local":
            ref_llm = get_local_llm(config.REF_MODEL, config.REF_GPU_MEM, config.REF_MAX_MODEL_LEN)
        else:
            ref_client = OpenAI(api_key=config.REF_API_KEY, base_url=config.REF_BASE_URL)
        self.refiner = VariantRefiner(mode=config.REF_MODE, client=ref_client, llm=ref_llm)

        # 4. Define Prompt Template
        self.PROMPT_TEMPLATE = """Your task is to generate a code implementation based on the provided pseudo-code logic. The implementation must strictly follow the specific style dimensions provided below.

### Target Style Configuration
{style_description}

### Output Format
You must output the code strictly using the <VERSION>, <INFO>, and <CODE> XML tags. Do not use markdown code blocks.

Example Output:
<VERSION>
<INFO>
Programming Paradigm: Procedural, Naming: snake_case ...
</INFO>
<CODE>
def my_function():
    pass
</CODE>
</VERSION>

### Task
Docstring:
{docstring}

Pseudo-code:
{pseudo_code}
"""

    def _count_tokens(self, text: str) -> int:
        """Helper to count tokens locally using tiktoken."""
        if not text or not self.encoding: return 0
        return len(self.encoding.encode(text))

    def extract_code_from_xml(self, text: str) -> Optional[str]:
        """Extract code from <CODE>...</CODE> tags."""
        match = re.search(r"<CODE>(.*?)</CODE>", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def prepare_batch_prompts(self, batch_data: List[Dict]) -> Tuple[List[str], List[List[Dict]], List[int]]:
        """Prepare flattened prompts and return them along with their input token counts."""
        all_prompts = []
        batch_styles_map = []
        prompt_token_counts = []
        
        for item in batch_data:
            docstring = item.get("docstring", item.get("doc", ""))
            pseudo_code = item.get("pseudo_code", "")
            styles = self.style_manager.get_diverse_styles(config.NUM_VARIANTS)
            batch_styles_map.append(styles)
            
            for style in styles:
                style_desc = self.style_manager.format_style_prompt(style)
                full_prompt = self.PROMPT_TEMPLATE.format(
                    style_description=style_desc,
                    docstring=docstring,
                    pseudo_code=pseudo_code
                )
                all_prompts.append(full_prompt)
                prompt_token_counts.append(self._count_tokens(full_prompt))
                
        return all_prompts, batch_styles_map, prompt_token_counts

    def generate_batch_vllm(self, prompts: List[str]) -> Tuple[List[str], List[int]]:
        """Batch generation using vLLM."""
        outputs = self.gen_llm.generate(prompts, self.gen_sampling_params, use_tqdm=False)
        results_content = []
        output_tokens_counts = []
        for output in outputs:
            text = output.outputs[0].text.strip()
            if "<CODE>" in text and "</CODE>" not in text:
                text += "</CODE></VERSION>"
            results_content.append(text)
            output_tokens_counts.append(len(output.outputs[0].token_ids))
        return results_content, output_tokens_counts

    def generate_batch_api(self, prompts: List[str]) -> Tuple[List[str], List[int]]:
        """Sequential generation using OpenAI API."""
        results_content = []
        output_tokens_counts = []
        for prompt in prompts:
            try:
                response = self.gen_client.chat.completions.create(
                    model=config.GEN_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.6,
                    max_tokens=config.GEN_MAX_TOKENS
                )
                text = response.choices[0].message.content.strip()
                results_content.append(text)
                output_tokens_counts.append(response.usage.completion_tokens)
            except Exception as e:
                logger.error(f"API Call failed: {e}")
                results_content.append("")
                output_tokens_counts.append(0)
        return results_content, output_tokens_counts

    def process_data_chunk(self, data_chunk: List[Dict], final_results_list: List, token_stats_list: List):
        """Unified processing with 6-dimension style strings and detailed token tracking."""
        if not data_chunk: return

        prompts, styles_map, input_tokens_list = self.prepare_batch_prompts(data_chunk)
        
        # Engine Selection
        if self.gen_mode == "local":
            logger.info(f"⚡ Batch Generating {len(prompts)} style variants via vLLM.")
            raw_outputs, output_toks = self.generate_batch_vllm(prompts)
        else:
            logger.info(f"⚡ Generating {len(prompts)} style variants via API.")
            raw_outputs, output_toks = self.generate_batch_api(prompts)
        
        current_idx = 0

        for i, item in enumerate(data_chunk):
            line_num = item.get("line_id", "unknown")
            docstring = item.get("docstring", item.get("doc", ""))
            pseudo_code = item.get("pseudo_code", "")
            item_styles = styles_map[i]
            
            flat_result = {
                "v1_code": None, "v1_info": None, "v2_code": None, "v2_info": None,
                "v3_code": None, "v3_info": None, "v4_code": None, "v4_info": None,
            }
            
            total_input_tokens = 0
            total_output_tokens = 0
            
            # --- ROW HEADER ---
            print("\n" + "═"*110)
            print(f" 📂 PROCESSING ROW: {line_num}")
            print(f" 📝 PSEUDO-CODE:\n{pseudo_code.strip()}")
            print("═"*110)

            for v_idx in range(config.NUM_VARIANTS):
                style = item_styles[v_idx]
                raw_text = raw_outputs[current_idx]
                
                # Update base tokens for initial generation
                total_input_tokens += input_tokens_list[current_idx]
                total_output_tokens += output_toks[current_idx]
                current_idx += 1
                
                variant_key = f"v{v_idx+1}"
                style_info_str = ", ".join([f"{k.capitalize()}: {v}" for k, v in style.items()])
                
                print(f"\n🚀 [{variant_key}] Target Style: {style_info_str}")
                
                code = self.extract_code_from_xml(raw_text)
                if not code:
                    print(f"   ❌ XML Extraction Failed for {variant_key}.")
                    continue

                # 1. Syntax Check & Refine
                syn_res = self.evaluator.check_runnability(code)
                if not syn_res["valid"]:
                    print(f"\n   [BEFORE REFINEMENT - SYNTAX ERROR]: {syn_res['error']}")
                    # Estimate tokens for refinement (input prompt + output code)
                    total_input_tokens += self._count_tokens(pseudo_code + code + str(style)) 
                    code = self.refiner.refine(pseudo_code, code, str(style), syn_res['error'], "syntax")
                    total_output_tokens += self._count_tokens(code)
                    print(f"   [AFTER REFINEMENT - FIXED SYNTAX]")
                else:
                    print(f"   ✅ Runnability Check: Passed")

                # 2. Compliance Check & Refine
                eval_res = self.evaluator.evaluate_compliance(pseudo_code, code, str(style))
                if not eval_res.get("is_valid", False):
                    feedback = eval_res.get("feedback")
                    print(f"\n   [BEFORE REFINEMENT - COMPLIANCE ISSUE]: {feedback}")
                    total_input_tokens += self._count_tokens(pseudo_code + code + feedback)
                    code = self.refiner.refine(pseudo_code, code, str(style), feedback, "compliance")
                    total_output_tokens += self._count_tokens(code)
                    print(f"   [AFTER REFINEMENT - FIXED COMPLIANCE]")
                else:
                    print(f"   ✅ Compliance Check: Passed")

                flat_result[f"{variant_key}_code"] = code
                flat_result[f"{variant_key}_info"] = style_info_str
                print(f"   ⭐ {variant_key} DONE")

            if any(flat_result.values()):
                res_item = item.copy()
                res_item.update(flat_result)
                final_results_list.append(res_item)
                
                # Token statistics per docstring (accumulated for all variants)
                token_stats_list.append({
                    "docstring": docstring,
                    "input_token": total_input_tokens,
                    "output_token": total_output_tokens
                })
            
            print("\n" + "╚" + "═"*108 + "╝")

    def run_batch(self):
        """Main execution flow"""
        file_counter = 1
        batch_data_buffer = []
        final_results = []
        token_stats = []
        
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        token_log = os.path.join(config.OUTPUT_DIR, "token_num_stage2.jsonl")

        with open(config.INPUT_FILE, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < config.START_LINE: continue
                if config.END_LINE and i > config.END_LINE: break
                
                try:
                    data = json.loads(line.strip())
                    data['line_id'] = i + 1
                    if "pseudo_code" in data and data["pseudo_code"]:
                        batch_data_buffer.append(data)
                    
                    limit = config.GEN_BATCH_SIZE if self.gen_mode == "local" else 1
                    
                    if len(batch_data_buffer) >= limit:
                        self.process_data_chunk(batch_data_buffer, final_results, token_stats)
                        batch_data_buffer = []
                        
                        if len(final_results) >= config.SAVE_INTERVAL:
                            self._save(final_results, file_counter)
                            self._save_tokens(token_stats, token_log)
                            final_results = []
                            token_stats = []
                            file_counter += 1
                            
                except Exception as e:
                    logger.error(f"Error at line {i+1}: {e}")
                    continue

            if batch_data_buffer:
                self.process_data_chunk(batch_data_buffer, final_results, token_stats)
            if final_results:
                self._save(final_results, file_counter)
                self._save_tokens(token_stats, token_log)

    def _save(self, data, idx):
        path = os.path.join(config.OUTPUT_DIR, f"variants_{idx}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"📁 Batch results saved to {path}")
        
    def _save_tokens(self, data, path):
        with open(path, "a", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    app = CodeVariantGenerator()
    app.run_batch()