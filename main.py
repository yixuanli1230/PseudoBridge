import json
import logging
import math
import os
import platform
import random
import re
import shutil
import sys
import torch
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime

from torch.utils.data import DataLoader
from sentence_transformers import (
    SentenceTransformer,
    models,
    losses,
    InputExample,
    LoggingHandler
)

# Import Configuration
from config import Config

class UnifiedModelTrainer:
    def __init__(self):
        self.config = Config()
        self.setup_logging()
        self.original_data_size = 0

    def setup_logging(self) -> None:
        """Configure basic logging settings."""
        logging.basicConfig(
            format="%(asctime)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO,
            handlers=[LoggingHandler()]
        )

    def print_device_info(self) -> None:
        """Print CUDA device information."""
        device_name = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using pytorch device: {device_name}")
        
        # Check Kernel Version
        min_recommended = (5, 5, 0)
        try:
            kernel_version = platform.release().split('-')[0]
            version_tuple = tuple(map(int, kernel_version.split('.')[:3]))
            if version_tuple < min_recommended:
                logging.warning(f"Kernel version {kernel_version} is below recommended {'.'.join(map(str, min_recommended))}.")
        except Exception:
            pass

    # =========================================================================
    # Data Loading Logic (Unified)
    # =========================================================================
    
    def load_data_stage1(self) -> List[InputExample]:
        """Load data for Stage 1: Requires 'docstring' and 'pseudo_code'."""
        all_data = []
        filtered_lines = []
        
        logging.info(f"Loading Stage 1 data from: {self.config.DATA_PATH}")
        
        with open(self.config.DATA_PATH, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, start=1):
                try:
                    data = json.loads(line)
                    # Stage 1: Align Docstring <-> Pseudo Code
                    docstring = data.get("docstring")
                    pseudo_code = data.get("pseudo_code")
                    
                    if not docstring or not pseudo_code:
                        filtered_lines.append(line_num)
                        continue

                    all_data.append([docstring, pseudo_code])

                except json.JSONDecodeError:
                    filtered_lines.append(line_num)
        
        self.original_data_size = len(all_data)
        
        # Sampling
        if self.config.SAMPLING_RATIO < 1.0:
            sample_size = int(len(all_data) * self.config.SAMPLING_RATIO)
            all_data = random.sample(all_data, sample_size)
            logging.info(f"Sampled {sample_size} examples ({self.config.SAMPLING_RATIO*100:.1f}%)")
        
        # Convert to InputExample
        return [InputExample(texts=[item[0], item[1]]) for item in all_data]

    def load_data_stage2(self, selected_labels: List[int] = None) -> List[InputExample]:
        """Load data for Stage 2: Requires 'docstring', 'code', and 'label'."""
        valid_samples = []
        label_distribution = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        
        logging.info(f"Loading Stage 2 data from: {self.config.DATA_PATH}")
        
        with open(self.config.DATA_PATH, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, start=1):
                try:
                    data = json.loads(line)
                    # Validation
                    if not all(key in data for key in ["docstring", "code", "label"]):
                        continue
                    if data["label"] not in [0, 1, 2, 3, 4]:
                        continue
                        
                    label = data["label"]
                    label_distribution[label] += 1
                    
                    # Stage 2: Align Docstring <-> Real Code (Filtered by Label)
                    if selected_labels is None or label in selected_labels:
                        valid_samples.append(
                            InputExample(texts=[data["docstring"], data["code"]])
                        )
                except json.JSONDecodeError:
                    continue
        
        # Logging stats
        logging.info("Label distribution:")
        for l, c in label_distribution.items():
            logging.info(f"Label {l}: {c}")
        
        self.original_data_size = len(valid_samples)
        return valid_samples

    # =========================================================================
    # Model Utils
    # =========================================================================

    def create_model(self, model_path: str) -> SentenceTransformer:
        """Create SentenceTransformer model."""
        logging.info(f"Creating model from: {model_path}")
        
        is_special_model = any(kw in model_path.lower() for kw in ["unixcode", "cocosoda"])
        
        word_embedding_model = models.Transformer(
            model_path,
            max_seq_length=1024 if is_special_model else None
        )
        
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=False,
            pooling_mode_cls_token=True,
            pooling_mode_max_tokens=False
        )
        return SentenceTransformer(modules=[word_embedding_model, pooling_model])

    def organize_output(self, output_path: str):
        """Clean up checkpoints, keep the latest."""
        try:
            output_path = Path(output_path)
            # Find all checkpoint directories
            checkpoint_dirs = sorted(
                [d for d in output_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
                key=lambda x: int(x.name.split("-")[1])
            )
            
            if not checkpoint_dirs:
                return

            latest_checkpoint = checkpoint_dirs[-1]
            
            # Move non-checkpoint files (like config.json, model.safetensors) from root to latest checkpoint
            # Note: SentenceTransformers saves files to root AND checkpoint folders usually.
            # Here we follow your logic: move root files to latest checkpoint folder? 
            # Or usually we keep the root clean. Let's stick to your logic of consolidating.
            
            # Implementation: Move everything from output_path to latest_checkpoint, excluding other checkpoints
            for item in output_path.iterdir():
                if item == latest_checkpoint: continue
                if item.is_dir() and item.name.startswith("checkpoint-"): continue # Don't move other checkpoints
                
                target = latest_checkpoint / item.name
                if target.exists():
                    if target.is_dir(): shutil.rmtree(str(target))
                    else: target.unlink()
                
                shutil.move(str(item), str(latest_checkpoint))
            
            logging.info(f"Consolidated files to {latest_checkpoint}")
            
        except Exception as e:
            logging.error(f"Error organizing output: {e}")

    # =========================================================================
    # Training Loop
    # =========================================================================

    def train(self, model_path: str, output_path: str, train_samples: List[InputExample]):
        """Core training function."""
        # 1. Prepare Model
        model = self.create_model(model_path)
        
        # 2. Prepare Loader
        train_dataloader = DataLoader(
            train_samples, 
            shuffle=True, 
            batch_size=self.config.TRAIN_BATCH_SIZE, 
            drop_last=True
        )
        
        # 3. Loss
        train_loss = losses.MultipleNegativesRankingLoss(model)
        
        # 4. Params
        warmup_steps = math.ceil(len(train_dataloader) * self.config.NUM_EPOCHS * self.config.WARMUP_RATIO)
        
        # 5. Output Directory
        if os.path.exists(output_path):
            # Careful with deletion, maybe backup? For now, we follow original logic
            try:
                shutil.rmtree(output_path)
            except: pass
        os.makedirs(output_path, exist_ok=True)
        
        logging.info(f"Output Directory: {output_path}")
        logging.info(f"Training parameters: Batch={self.config.TRAIN_BATCH_SIZE}, LR={self.config.LEARNING_RATE}, Epochs={self.config.NUM_EPOCHS}")

        # 6. Fit
        try:
            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=self.config.NUM_EPOCHS,
                warmup_steps=warmup_steps,
                optimizer_params={"lr": self.config.LEARNING_RATE},
                output_path=output_path,
                checkpoint_path=output_path,
                checkpoint_save_steps=self.config.SAVE_STEPS,
                use_amp=self.config.USE_AMP,
                show_progress_bar=True
            )
            
            # 7. Organize
            self.organize_output(output_path)
            logging.info(f"Training completed for {os.path.basename(model_path)}")
            
        except Exception as e:
            logging.error(f"Training failed: {e}")
            raise

    # =========================================================================
    # Stage 1 Workflow
    # =========================================================================
    
    def run_stage_1(self):
        """
        Stage 1: Base Model -> Train with (Docstring, Pseudo-code)
        """
        logging.info(">>> Starting STAGE 1 Training Workflow")
        
        # 1. Scan Base Models
        base_dir = Path(self.config.BASE_MODELS_PATH)
        if not base_dir.exists():
            logging.error(f"Base models directory not found: {base_dir}")
            return

        models_list = [d.name for d in base_dir.iterdir() if d.is_dir()]
        models_map = {i+1: name for i, name in enumerate(models_list)}
        
        # 2. Interactive Selection
        print("\nAvailable Base Models:")
        for idx, name in models_map.items():
            print(f"{idx}. {name}")
            
        selection = input("\nEnter model numbers (comma-separated) or 'all': ")
        if selection.lower() == 'all':
            selected_models = list(models_map.values())
        else:
            indices = [int(s.strip()) for s in selection.split(',') if s.strip().isdigit()]
            selected_models = [models_map[i] for i in indices if i in models_map]

        if not selected_models:
            logging.warning("No models selected.")
            return

        # 3. Load Data
        train_samples = self.load_data_stage1()
        if not train_samples:
            logging.error("No training data loaded.")
            return

        # 4. Train Loop
        for model_name in selected_models:
            input_model_path = os.path.join(self.config.BASE_MODELS_PATH, model_name)
            
            # Create Output Path: .../ModelName/train_pseudo_code_1.0/
            # Remove suffixes like -base, -instruct for cleaner folder names
            clean_name = re.sub(r'(-base|-large|-small|-instruct)?$', '', model_name, flags=re.IGNORECASE)
            model_out_dir = os.path.join(self.config.STAGE1_OUTPUT_DIR, clean_name)
            output_path = os.path.join(model_out_dir, f"train_pseudo_code_{self.config.SAMPLING_RATIO:.1f}")

            self.train(input_model_path, output_path, train_samples)

    # =========================================================================
    # Stage 2 Workflow
    # =========================================================================

    def run_stage_2(self):
        """
        Stage 2: Previous Checkpoint -> Train with (Docstring, Code) filtered by Label
        """
        logging.info(">>> Starting STAGE 2 Training Workflow")

        # 1. Select Labels
        print("\nSelect labels to train on (e.g., '3,4' for high quality):")
        print("0-4 available. Enter 'all' for all labels.")
        choice = input("Selection: ")
        if choice.lower() == 'all':
            selected_labels = None
        else:
            selected_labels = [int(c.strip()) for c in choice.split(",") if c.strip().isdigit()]
            logging.info(f"Selected labels: {selected_labels}")

        # 2. Load Data
        train_samples = self.load_data_stage2(selected_labels)
        if not train_samples:
            logging.error("No training data loaded.")
            return

        # 3. Scan Previous Stage Outputs
        root_scan_dir = self.config.STAGE2_INPUT_DIR
        logging.info(f"Scanning for Stage 1 models in: {root_scan_dir}")
        
        if not os.path.exists(root_scan_dir):
            logging.error("Stage 1 output directory does not exist.")
            return

        # Structure: ModelName -> StepDir (train_pseudo_code_1.0) -> CheckpointDir
        available_checkpoints = []
        
        for model_type in os.listdir(root_scan_dir):
            model_path = os.path.join(root_scan_dir, model_type)
            if not os.path.isdir(model_path): continue
            
            for step_dir in os.listdir(model_path):
                step_path = os.path.join(model_path, step_dir)
                if not os.path.isdir(step_path): continue
                
                # Find latest checkpoint
                checkpoints = [d for d in os.listdir(step_path) if d.startswith("checkpoint-")]
                if checkpoints:
                    checkpoints.sort(key=lambda x: int(x.split("-")[1]))
                    latest = checkpoints[-1]
                    full_ckpt_path = os.path.join(step_path, latest)
                    available_checkpoints.append({
                        "name": f"{model_type} / {step_dir} / {latest}",
                        "path": full_ckpt_path,
                        "base_model_name": model_type,
                        "prev_step_name": step_dir
                    })

        if not available_checkpoints:
            logging.error("No Stage 1 checkpoints found.")
            return

        # 4. Interactive Selection
        print("\nAvailable Stage 1 Checkpoints:")
        for i, ckpt in enumerate(available_checkpoints):
            print(f"{i+1}. {ckpt['name']}")
            
        selection = input("\nEnter numbers (comma-separated) or 'all': ")
        if selection.lower() == 'all':
            selected_ckpts = available_checkpoints
        else:
            indices = [int(s.strip())-1 for s in selection.split(',') if s.strip().isdigit()]
            selected_ckpts = [available_checkpoints[i] for i in indices if 0 <= i < len(available_checkpoints)]

        # 5. Train Loop
        for ckpt in selected_ckpts:
            input_path = ckpt['path']
            
            # Construct Stage 2 Output Path
            # e.g., .../ast_ablation/ModelName/train_pseudo_code_1.0_step2
            model_name = ckpt['base_model_name']
            step_name = ckpt['prev_step_name'] # e.g. train_pseudo_code_1.0
            
            output_dir_name = f"{step_name}_step2"
            output_path = os.path.join(self.config.STAGE2_OUTPUT_DIR, model_name, output_dir_name)

            self.train(input_path, output_path, train_samples)

    # =========================================================================
    # Entry Point
    # =========================================================================
    
    def run(self):
        self.print_device_info()
        
        if self.config.STAGE == 1:
            self.run_stage_1()
        elif self.config.STAGE == 2:
            self.run_stage_2()
        else:
            logging.error(f"Invalid STAGE configuration: {self.config.STAGE}")

if __name__ == "__main__":
    trainer = UnifiedModelTrainer()
    trainer.run()