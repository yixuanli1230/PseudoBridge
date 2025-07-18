from transformers import RobertaModel, RobertaTokenizer, T5EncoderModel
from sentence_transformers import SentenceTransformer
import json
import torch
import numpy as np
import os
import random
from typing import Tuple, List, Dict, Optional
import torch.nn.functional as F

class EvaluationConfig:
    def __init__(self, batch_size=64, start_line=0, end_line=None, sample_size=None):
        self.batch_size = batch_size
        self.start_line = start_line
        self.end_line = end_line
        self.sample_size = sample_size

MODEL_MAP = {
    "sentencebert": SentenceTransformer,
    "roberta": (RobertaModel, RobertaTokenizer),
    "codet5": (T5EncoderModel, RobertaTokenizer)
}

class CodeSearchEvaluator:
    def __init__(self, model_path: str, model_type: str):
        self.model_path = model_path
        self.model_type = model_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if model_type == "sentencebert":
            self.model = MODEL_MAP[model_type](model_path)
            if any(kw in model_path.lower() for kw in ["unixcode", "cocosoda"]):
                self.model.max_seq_length = 1024
        else:
            model_class, tokenizer_class = MODEL_MAP[model_type]
            self.tokenizer = tokenizer_class.from_pretrained(model_path)
            self.model = model_class.from_pretrained(model_path)
            self.model.eval()
            self.max_length = 512
        
        self.model.to(self.device)

    def load_jsonl_data(self, file_path: str, config: EvaluationConfig) -> Tuple[List[str], List[str]]:
        valid_lines = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i < config.start_line: continue
                if config.end_line and i >= config.end_line: break
                if line.strip(): valid_lines.append(line.strip())
        
        if config.sample_size and config.sample_size < len(valid_lines):
            valid_lines = random.sample(valid_lines, config.sample_size)
        
        queries, codes = [], []
        for line in valid_lines:
            try:
                data = json.loads(line)
                queries.append(data['docstring'])
                codes.append(data['code'])
            except (json.JSONDecodeError, KeyError):
                pass
        
        return queries, codes
        
    def compute_embeddings(self, texts: List[str], batch_size: int) -> np.ndarray:
        if self.model_type == "sentencebert":
            return self.model.encode(texts, batch_size=batch_size, device=self.device)
        
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                mask = inputs.attention_mask.unsqueeze(-1)
                hidden = outputs.last_hidden_state
                pooled = (hidden * mask).sum(1) / mask.sum(1)
                embeddings.append(pooled.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def evaluate(self, file_path: str, config: EvaluationConfig) -> Dict[str, float]:
        queries, codes = self.load_jsonl_data(file_path, config)
        if not queries or not codes:
            return {"error": "No valid data loaded"}
        
        q_emb = self.compute_embeddings(queries, config.batch_size)
        c_emb = self.compute_embeddings(codes, config.batch_size)
        
        Q = torch.tensor(q_emb).to(self.device)
        C = torch.tensor(c_emb).to(self.device)
        Q_norm = F.normalize(Q, p=2, dim=1)
        C_norm = F.normalize(C, p=2, dim=1)
        
        sim = torch.mm(Q_norm, C_norm.t())
        indices = torch.argsort(sim, dim=1, descending=True)
        
        ranks = [(indices[i] == i).nonzero().item() + 1 for i in range(len(Q))]
        return {
            'mrr': sum(1/r for r in ranks) / len(ranks),
            'recall@1': sum(r <= 1 for r in ranks) / len(ranks),
            'recall@3': sum(r <= 3 for r in ranks) / len(ranks),
            'recall@5': sum(r <= 5 for r in ranks) / len(ranks),
            'avg_rank': sum(ranks) / len(ranks),
            'samples': len(ranks)
        }


def run_evaluation(model_path: str, data_path: str, model_type: str):
    config = EvaluationConfig(batch_size=64)
    evaluator = CodeSearchEvaluator(model_path, model_type)
    metrics = evaluator.evaluate(data_path, config)
    
    print("\nEvaluation Results:")
    print(f"Model: {os.path.basename(model_path)}")
    print(f"Dataset: {os.path.basename(data_path)}")
    if 'error' in metrics:
        print(f"Error: {metrics['error']}")
    else:
        print(f"Samples: {metrics['samples']}")
        print(f"MRR: {metrics['mrr']:.4f}")
        print(f"Recall@1: {metrics['recall@1']:.4f}")
        print(f"Recall@3: {metrics['recall@3']:.4f}")
        print(f"Recall@5: {metrics['recall@5']:.4f}")
        print(f"Avg Rank: {metrics['avg_rank']:.2f}")


if __name__ == '__main__':
    model_path = "/path/to/model"
    data_path = "/path/to/dataset.jsonl"
    model_type = "roberta"  # "sentencebert", "roberta", or "codet5"
    
    run_evaluation(model_path, data_path, model_type)