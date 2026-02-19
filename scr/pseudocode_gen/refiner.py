import logging
import re
from openai import OpenAI
import config

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None

logger = logging.getLogger(__name__)

class PseudoCodeRefiner:
    def __init__(self):
        self.mode = config.REF_MODE
        self.model_name = config.REF_MODEL

        # Initialize Refiner's own model instance
        if self.mode == "local":
            if LLM is None:
                raise ImportError("vLLM is required for local mode. Please install vllm.")
            logger.info(f"🛠️ Loading Refiner Local Model: {self.model_name}...")
            self.llm = LLM(
                model=self.model_name,
                gpu_memory_utilization=config.REF_GPU_MEM,
                max_model_len=config.REF_MAX_MODEL_LEN,
                trust_remote_code=True
            )
            self.sampling_params = SamplingParams(
                temperature=0.6,
                max_tokens=config.REF_MAX_MODEL_LEN
            )
        else:
            self.client = OpenAI(api_key=config.REF_API_KEY, base_url=config.REF_BASE_URL)

        self.STYLE_GUIDELINES = """ You are a professional Python developer and now need to convert the given docstring content into standardised pseudo-code.
        1. Comments: use `//` for single line comments s, but do not use multi-line comments.
        2. Variables: Choose clear typed names for variables.
        3. Input/Output: Keep input/output simple and clear.
        4. Conditional Clauses: Use `IF`, `ELSIF` and `ELSE` with proper indentation.
        5. Loops: Use `FOR`, `WHILE` or `DO.... .WHILE` loops, specify conditions and indent the code.
        6. Functions/Procedures: name functions descriptively and consider arguments. 
        7. Formatting: Maintain consistent 2-4 space indentation for clarity.
        8. Content: Give only the key steps and ignore details.
        """

    def refine(self, docstring: str, original_code: str, old_pseudo_code: str, feedback: str, issue_type: str = "quality") -> str:
        """Refine the pseudo-code based on the feedback."""
        if issue_type == "alignment":
            instruction = "CRITICAL: The previous pseudo-code failed the LOGIC ALIGNMENT check. Fix the logic to match the Original Code."
        else:
            instruction = "The previous pseudo-code logic is correct but QUALITY is low. Improve it based on the 5-dimension standards."

        prompt = f"""
        You are a professional Python developer. 
        {instruction}

        ### Specific Feedback to Fix:
        {feedback}

        ### Context
        [Docstring]: {docstring}
        [Original Code]: {original_code}
        [Current Pseudo-code]: {old_pseudo_code}

        ### Refinement Requirements (STRICT):
        1. Fix the issues mentioned in the Feedback above.
        2. Follow these Style Guidelines:
           {self.STYLE_GUIDELINES}
        3. Output Format: Output ONLY the raw pseudo-code text without markdown blocks.
        """

        if self.mode == "local":
            outputs = self.llm.generate([prompt], self.sampling_params, use_tqdm=False)
            raw_content = outputs[0].outputs[0].text.strip()
        else:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    stream=False,
                    temperature=0.6,
                    max_tokens=config.REF_MAX_TOKENS 
                )
                raw_content = response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"Refiner API failed: {e}")
                return old_pseudo_code

        return self._clean_output(raw_content)

    def _clean_output(self, text: str) -> str:
        """Clear any unwanted markdown formatting."""
        text = re.sub(r'^```[a-zA-Z]*\n', '', text)
        text = re.sub(r'\n```$', '', text)
        return text.strip()