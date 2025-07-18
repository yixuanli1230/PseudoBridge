import json
import logging
import math
import os
from typing import List, Tuple, Dict, Any, Optional

from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, models, losses, InputExample, LoggingHandler

class ModelTrainer:
    def __init__(self):
        logging.basicConfig(
            format="%(asctime)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO,
            handlers=[LoggingHandler()]
        )

    def validate_line(self, line: str) -> Optional[Dict[str, Any]]:
        try:
            data = json.loads(line)
            required = ["docstring", "pseudo_code", "code", "label"]
            if not all(key in data for key in required):
                return None
            if data["label"] not in [0, 1, 2, 3, 4]:
                return None
            return data
        except json.JSONDecodeError:
            return None

    def load_data(self, file_path: str, selected_labels: List[int] = None) -> Tuple[List[InputExample], Dict[int, int]]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        valid_samples = []
        invalid_count = 0
        label_dist = {i: 0 for i in range(5)}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = self.validate_line(line)
                if not data:
                    invalid_count += 1
                    continue
                
                label = data["label"]
                label_dist[label] += 1
                if selected_labels is None or label in selected_labels:
                    valid_samples.append(InputExample(texts=[data["docstring"], data["code"]]))
        
        logging.info(f"Loaded {len(valid_samples)} samples (Invalid: {invalid_count})")
        logging.info(f"Label distribution: {label_dist}")
        return valid_samples, label_dist

    def create_model(self, model_path: str) -> SentenceTransformer:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        word_embedding = models.Transformer(model_path)
        pooling = models.Pooling(
            word_embedding.get_word_embedding_dimension(),
            pooling_mode_cls_token=True
        )
        return SentenceTransformer(modules=[word_embedding, pooling])

    def train_model(self, model_path: str, output_path: str, data_path: str, selected_labels: List[int] = None):
        try:
            train_samples, label_dist = self.load_data(data_path, selected_labels)
            model = self.create_model(model_path)
            
            dataloader = DataLoader(train_samples, shuffle=True, batch_size=48, drop_last=True)
            loss = losses.MultipleNegativesRankingLoss(model)

            epochs = 3
            warmup_steps = math.ceil(len(dataloader) * epochs * 0.1)
            
            os.makedirs(output_path, exist_ok=True)
            logging.info(f"Training for {epochs} epochs (warmup: {warmup_steps} steps)")
            logging.info(f"Training model: {model_path}")
            logging.info(f"Output path: {output_path}")
            logging.info(f"Using labels: {selected_labels or 'All'}")
            
            model.fit(
                train_objectives=[(dataloader, loss)],
                epochs=epochs,
                warmup_steps=warmup_steps,
                optimizer_params={"lr": 5e-5},
                checkpoint_path=output_path,
                checkpoint_save_steps=3000,
                use_amp=False
            )
            logging.info(f"Model saved to: {output_path}")

        except Exception as e:
            logging.error(f"Training failed: {str(e)}")
            raise

def main():
    DATA_PATH = "/path/to/dataset.jsonl"            # Update with your dataset path
    BASE_MODEL_PATHS = [                            # Update with your model paths
        "/path/to/model1",
        "/path/to/model2",
    ]
    OUTPUT_DIR = "/path/to/output"                  # Update with output directory
    SELECTED_LABELS = None                          # None for all labels, or specify e.g. [0,1,2]
    
    trainer = ModelTrainer()
    
    for model_path in BASE_MODEL_PATHS:
        model_name = os.path.basename(model_path)
        output_path = os.path.join(OUTPUT_DIR, f"{model_name}-finetuned")
        trainer.train_model(model_path, output_path, DATA_PATH, SELECTED_LABELS)

if __name__ == "__main__":
    main()