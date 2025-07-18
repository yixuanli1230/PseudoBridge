import json
import logging
import math
import os
import random
import re
import shutil
import torch
from pathlib import Path
from typing import List
from sentence_transformers import SentenceTransformer, models, losses, InputExample, LoggingHandler
from sentence_transformers.losses import MultipleNegativesRankingLoss
from torch.utils.data import DataLoader

class ModelTrainer:
    def __init__(self):
        self.setup_logging()
        self.config = {
            'train_batch_size': 48,
            'num_epochs': 3,
            'learning_rate': 5e-5,
            'warmup_ratio': 0.1,
            'use_amp': False
        }
        self.original_data_size = 0
        self.logger = logging.getLogger(__name__)

    def setup_logging(self) -> None:
        logging.basicConfig(
            format="%(asctime)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO,
            handlers=[LoggingHandler()]
        )
    
    def log(self, msg: str) -> None:
        self.logger.info(msg)

    def load_and_filter_data(self, data_path: str, sampling_ratio: float = 1.0) -> List[list]:
        """Load and sample data from JSONL file"""
        valid_data = []
        with open(data_path) as f:
            for i, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    if all(key in data for key in ("docstring", "pseudo_code", "code")):
                        valid_data.append([data["docstring"], data["pseudo_code"], data["code"]])
                except json.JSONDecodeError:
                    self.log(f"JSON decode error at line {i}")

        self.original_data_size = len(valid_data)
        if sampling_ratio < 1.0:
            sample_size = int(len(valid_data) * sampling_ratio)
            valid_data = random.sample(valid_data, sample_size)
            self.log(f"Sampled {sample_size} examples ({sampling_ratio*100:.1f}% of data)")
        
        return valid_data

    def create_model(self, model_path: str) -> SentenceTransformer:
        """Initialize SentenceTransformer model"""
        max_seq_length = 512 if any(k in model_path.lower() for k in ("unixcode", "cocosoda")) else None
        word_embedding = models.Transformer(model_path, max_seq_length=max_seq_length)
        pooling = models.Pooling(
            word_embedding.get_word_embedding_dimension(),
            pooling_mode_cls_token=True
        )
        return SentenceTransformer(modules=[word_embedding, pooling])

    def train_single_model(self, base_models_path: str, data_path: str, 
                         base_output_path: str, model_name: str, sampling_ratio: float) -> None:
        """Train a single model end-to-end"""
        full_model_path = Path(base_models_path) / model_name
        output_path = Path(base_output_path) / re.sub(r'(-base|-large|-small)?$', '', model_name, flags=re.I)
        output_path = output_path / f"train_pseudo_code_{sampling_ratio:.1f}"
        
        shutil.rmtree(output_path, ignore_errors=True)
        output_path.mkdir(parents=True, exist_ok=True)

        # Setup and logging
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.log(f"Using device: {device}")
        self.log(f"Training model: {model_name}\nOutput dir: {output_path}\nSampling ratio: {sampling_ratio}")

        # Data preparation
        data = self.load_and_filter_data(data_path, sampling_ratio)
        train_samples = [InputExample(texts=[doc, pseudo]) for doc, pseudo, _ in data]
        self.log(f"Training samples: {len(train_samples)}/{self.original_data_size}")

        # Model initialization
        model = self.create_model(str(full_model_path))
        train_loader = DataLoader(
            train_samples, 
            shuffle=True, 
            batch_size=self.config['train_batch_size'], 
            drop_last=True
        )
        loss = MultipleNegativesRankingLoss(model)
        warmup_steps = math.ceil(len(train_loader) * self.config['num_epochs'] * self.config['warmup_ratio'])

        # Training
        model.fit(
            train_objectives=[(train_loader, loss)],
            epochs=self.config['num_epochs'],
            warmup_steps=warmup_steps,
            optimizer_params={"lr": self.config['learning_rate']},
            output_path=str(output_path),
            use_amp=self.config['use_amp'],
            show_progress_bar=True
        )
        self.log(f"Training completed. Model saved to: {output_path}")

def main():
    DATA_PATH = "/path/to/your/data.jsonl"  
    ROOT_DIR = "/path/to/root"             
    
    MODEL_NAMES = [
        "model-name-1",
        "model-name-2",
        # you can add more models
    ]
    
    SAMPLING_RATIO = 1.0  
    
    models_path = Path(ROOT_DIR) / "Models"
    output_path = Path(ROOT_DIR) / "Finetune_models" / "stage1"
    
    trainer = ModelTrainer()
    
    for model_name in MODEL_NAMES:
        trainer.train_single_model(
            base_models_path=str(models_path),
            data_path=DATA_PATH,
            base_output_path=str(output_path),
            model_name=model_name,
            sampling_ratio=SAMPLING_RATIO
        )

if __name__ == "__main__":
    main()