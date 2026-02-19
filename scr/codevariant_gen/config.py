import os

# ==========================================
# 1. General Pipeline Settings
# ==========================================
INPUT_FILE = "results/input.jsonl"
OUTPUT_DIR = "results/output"
FINAL_OUTPUT_FILE = "results/final_output.jsonl"
SAVE_INTERVAL = 100
START_LINE = 1
END_LINE = None
MAX_RETRIES = 3
NUM_VARIANTS = 4

# ==========================================
# 2. Generator Settings
# ==========================================
# Mode: "api" (OpenAI API) or "local" (vLLM in-memory)
GEN_MODE = "api" 
GEN_MODEL = "gpt-4o-mini" # Use model name for API, or absolute path for local
GEN_BASE_URL = None 
GEN_API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")

# Local mode specific settings
GEN_MAX_MODEL_LEN = 8192      # Context window limit for local model
GEN_GPU_MEM = 0.4             # GPU memory utilization
GEN_BATCH_SIZE = 16           # vLLM inference batch size

# API mode specific settings
GEN_MAX_TOKENS = 1024         # Max output tokens for API calls

# ==========================================
# 3. Refiner Settings
# ==========================================
REF_MODE = "api" 
REF_MODEL = "gpt-4o-mini" 
REF_BASE_URL = None 
REF_API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")

# Local mode specific settings
REF_MAX_MODEL_LEN = 8192      
REF_GPU_MEM = 0.3             
REF_BATCH_SIZE = 16           

# API mode specific settings
REF_MAX_TOKENS = 2048         

# ==========================================
# 4. Evaluator Settings
# ==========================================
EVAL_MODE = "api"
EVAL_MODEL = "gpt-4-turbo"
EVAL_BASE_URL = None
EVAL_API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")

# Local mode specific settings
EVAL_MAX_MODEL_LEN = 8192
EVAL_GPU_MEM = 0.2

# API mode specific settings
EVAL_MAX_TOKENS = 512         # Evaluation results are short JSONs