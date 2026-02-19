import os

class Config:
    # =========================================================================
    # 1. Global Configuration (全局配置)
    # =========================================================================
    # Training Stage: 1 or 2
    # Stage 1: Align Docstring <-> Pseudo-code (Base Model -> Finetuned Step 1)
    # Stage 2: Align Docstring <-> Code with Labels (Finetuned Step 1 -> Finetuned Step 2)
    STAGE = 1
    
    # Root directory of your project
    # Ensure this path contains "2_Models" and "3_Finetune_models" folders
    ROOT_DIR = "/cpfs01/projects-HDD/cfff-0082a359858b_HDD/lyx_24110240175/000_PseudoBridge"

    # =========================================================================
    # 2. Data Configuration (数据配置)
    # =========================================================================
    # Path to the JSONL dataset
    # For Stage 1: Should contain "docstring" and "pseudo_code"
    # For Stage 2: Should contain "docstring", "code", and "label"
    DATA_PATH = "/cpfs01/projects-HDD/cfff-0082a359858b_HDD/lyx_24110240175/000_PseudoBridge/4_Data/new_gen_26.01.17/temp_dir/qwen2.5_7b_coder_instruct_step1.jsonl"
    
    # Sampling ratio (0.0 - 1.0) to use a subset of data
    SAMPLING_RATIO = 1.0

    # =========================================================================
    # 3. Hyperparameters (超参数)
    # =========================================================================
    TRAIN_BATCH_SIZE = 48
    NUM_EPOCHS = 3
    LEARNING_RATE = 5e-5
    WARMUP_RATIO = 0.1
    USE_AMP = False         # Mixed Precision Training
    
    # Checkpoint saving frequency (steps)
    SAVE_STEPS = 3000

    # =========================================================================
    # 4. Path Management (Auto-generated based on ROOT_DIR)
    # =========================================================================
    @property
    def BASE_MODELS_PATH(self):
        return os.path.join(self.ROOT_DIR, "models/base")

    @property
    def STAGE1_OUTPUT_DIR(self):
        # Output directory for Stage 1 results
        return os.path.join(self.ROOT_DIR, "models/finetuned", "stage1")

    @property
    def STAGE2_INPUT_DIR(self):
        # Usually Stage 2 reads models from Stage 1 output
        return self.STAGE1_OUTPUT_DIR

    @property
    def STAGE2_OUTPUT_DIR(self):
        # Output directory for Stage 2 results
        return os.path.join(self.ROOT_DIR, "models/finetuned", "stage2")