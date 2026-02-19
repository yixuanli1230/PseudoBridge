import json
import logging
import ast
from typing import Optional
from openai import OpenAI
import config

logger = logging.getLogger(__name__)

class VariantEvaluator:
    def __init__(self, mode: str, client: Optional[OpenAI] = None, llm=None):
        self.mode = mode
        self.client = client
        self.llm = llm

        if self.mode == "local" and self.llm is not None:
            try:
                from vllm import SamplingParams
                self.sampling_params = SamplingParams(
                    temperature=0.1,  # Use low temperature for consistent evaluation
                    max_tokens=config.EVAL_MAX_TOKENS
                )
            except ImportError:
                pass

    def check_runnability(self, code: str) -> dict:
        """
        Check if the generated code is syntactically correct and compilable.
        
        Since we do not have specific test cases/inputs for every snippet, 
        we use `compile()` as a proxy for 'runnability'. If it compiles, 
        it is free of syntax errors and fundamental structure issues.
        """
        try:
            # compile() is stricter than ast.parse(); it generates bytecode.
            # mode='exec' checks if the code block can be executed as a module.
            compile(code, filename="<string>", mode="exec")
            return {"valid": True, "error": None}
        except SyntaxError as e:
            return {"valid": False, "error": f"SyntaxError: {str(e)}"}
        except Exception as e:
            return {"valid": False, "error": f"Compilation Error: {str(e)}"}

    def _extract_json(self, text: str) -> dict:
        """Helper to safely extract JSON from raw model output."""
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end != 0:
                return json.loads(text[start:end])
            return {"is_valid": False, "feedback": "No valid JSON found in output."}
        except Exception as e:
            return {"is_valid": False, "feedback": f"JSON Parse Error: {str(e)}"}

    def evaluate_compliance(self, pseudo_code: str, generated_code: str, style_requirements: str) -> dict:
        """
        Use LLM to evaluate if the code aligns functionally with the pseudo-code
        and adheres to the specified style requirements.
        """
        prompt = f"""
        You are a Senior Code Reviewer. Your task is to evaluate the "Generated Code" against the "Pseudo-code" and "Style Requirements".
        
        ### Evaluation Steps:
        1. **Functional Alignment (Critical)**: 
           - Does the generated code implement the EXACT logic described in the pseudo-code? 
           - Check for missing steps, wrong conditions, or hallucinated logic.
           
        2. **Style Compliance**: 
           - Does the code strictly follow the provided Style Requirements (e.g., Naming, Paradigm, Error Handling)?

        ### Inputs:
        [Pseudo-code]:
        {pseudo_code}

        [Style Requirements]:
        {style_requirements}

        [Generated Code]:
        {generated_code}

        ### Output Format:
        Return a JSON object with the following fields:
        - "logic_aligned": boolean (True if logic is correct)
        - "style_compliant": boolean (True if style is correct)
        - "is_valid": boolean (True ONLY if BOTH logic_aligned AND style_compliant are True)
        - "feedback": string (Concise feedback. If invalid, explain why. If valid, say "Looks good".)

        Output JSON only:
        """
        
        try:
            if self.mode == "local":
                outputs = self.llm.generate([prompt], self.sampling_params, use_tqdm=False)
                raw_text = outputs[0].outputs[0].text.strip()
                result = self._extract_json(raw_text)
            else:
                response = self.client.chat.completions.create(
                    model=config.EVAL_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
                result = json.loads(response.choices[0].message.content)
            
            # Fallback keys to ensure robustness
            if "is_valid" not in result:
                result["is_valid"] = result.get("logic_aligned", False) and result.get("style_compliant", False)
                
            return result
            
        except Exception as e:
            logger.error(f"Compliance evaluation failed: {e}")
            return {
                "is_valid": False, 
                "logic_aligned": False, 
                "style_compliant": False, 
                "feedback": f"Evaluator Error: {str(e)}"
            }