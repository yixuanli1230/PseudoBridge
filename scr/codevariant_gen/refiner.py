import logging
import re
from typing import Optional
from openai import OpenAI
import config

logger = logging.getLogger(__name__)

class VariantRefiner:
    def __init__(self, mode: str, client: Optional[OpenAI] = None, llm=None):
        self.mode = mode
        self.client = client
        self.llm = llm

    def refine(self, pseudo_code: str, old_code: str, style_requirements: str, feedback: str, issue_type: str = "compliance") -> str:
        """
        Refine the code based on feedback.
        
        Args:
            issue_type: "syntax" (for compilation errors) or "compliance" (for logic/style mismatches)
        """
        
        # 1. Dynamically adjust System Instruction based on error type
        if issue_type == "syntax":
            instruction = "CRITICAL: The previous code has a SYNTAX ERROR and cannot run. Fix the syntax error while preserving the logic and style."
            temperature = 0.2 # Syntax fixes require low creativity
        else:
            instruction = "The previous code failed the Style/Logic check. Refactor the code to strictly match the [Style Requirements] and [Pseudo-code] logic."
            temperature = 0.6 # Style refactoring allows for some structural adjustments
            
        prompt = f"""
        You are a Python Expert. 
        {instruction}
        
        ### Issue to Fix:
        {feedback}

        ### Context
        [Pseudo-code]:
        {pseudo_code}

        [Style Requirements]:
        {style_requirements}

        [Current Code (To be fixed)]:
        {old_code}

        ### Output Requirements:
        1. Output ONLY the raw Python code.
        2. Do NOT use markdown code blocks (no ```).
        3. Do NOT add explanations or comments outside the code.
        """
        
        try:
            if self.mode == "local":
                from vllm import SamplingParams
                sampling_params = SamplingParams(temperature=temperature, max_tokens=config.REF_MAX_TOKENS)
                outputs = self.llm.generate([prompt], sampling_params, use_tqdm=False)
                raw_content = outputs[0].outputs[0].text.strip()
            else:
                response = self.client.chat.completions.create(
                    model=config.REF_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    stream=False,
                    temperature=temperature
                )
                raw_content = response.choices[0].message.content.strip()
                
            return self._extract_code(raw_content)
            
        except Exception as e:
            logger.error(f"Refiner failed: {e}")
            return old_code

    def _extract_code(self, text: str) -> str:
        """
        More robust code extraction logic:
        1. Try to extract content within Markdown code blocks (```...```).
        2. If no code block, remove the starting and ending ``` tags (backward compatibility).
        3. If it's plain text, return it directly.
        """
        # Attempt to match ```python ... ``` or ``` ... ```
        # re.DOTALL allows '.' to match newlines
        match = re.search(r"```[a-zA-Z]*\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # If no paired ``` are found, handle edge cases with only start or end tags (defensive)
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
        
        return text.strip()