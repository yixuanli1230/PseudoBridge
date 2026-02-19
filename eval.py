"""
Evaluates SentenceTransformer-based models on code retrieval tasks using MRR and Recall@k metrics.
Supports both base models and fine-tuned checkpoints.
"""

import json
import logging
import random
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

@dataclass
class EvalConfig:
    """Configuration for evaluation parameters."""
    batch_size: int = 64
    start_line: int = 0
    end_line: Optional[int] = None
    sample_size: Optional[int] = None  # None means use all data

class CodeSearchEvaluator:
    def __init__(self, model_path: Union[str, Path]):
        self.model_path = Path(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading model from {self.model_path} on {self.device}...")
        self.model = SentenceTransformer(str(self.model_path))
        self.model.to(self.device)

        # Special configuration for specific models
        model_name_lower = self.model_path.name.lower()
        if any(kw in model_name_lower for kw in ["unixcode", "cocosoda"]):
            self.model.max_seq_length = 1024

    def load_data(self, file_path: Path, config: EvalConfig) -> Tuple[List[str], List[str]]:
        """Loads and parses JSONL data containing 'docstring' and 'code' fields."""
        queries, codes = [], []
        valid_lines = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Read all lines first to handle indexing/slicing easier
                lines = f.readlines()
                
            # Slice based on config
            end = config.end_line if config.end_line else len(lines)
            lines = lines[config.start_line : end]
            
            # Filter empty lines
            valid_lines = [line.strip() for line in lines if line.strip()]
            
            # Sampling
            if config.sample_size and config.sample_size < len(valid_lines):
                valid_lines = random.sample(valid_lines, config.sample_size)
            
            # Parse
            skipped = 0
            for line in valid_lines:
                try:
                    data = json.loads(line)
                    queries.append(data['docstring'])
                    codes.append(data['code'])
                except (json.JSONDecodeError, KeyError):
                    skipped += 1
            
            if skipped > 0:
                logger.warning(f"Skipped {skipped} malformed lines.")
            
            logger.info(f"Loaded {len(queries)} pairs from {file_path.name}")
            return queries, codes

        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return [], []

    def compute_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Computes embeddings for a list of texts."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            device=self.device,
            convert_to_numpy=True,
            normalize_embeddings=True # Usually recommended for cosine similarity
        )

    def evaluate(self, queries_emb: np.ndarray, codes_emb: np.ndarray) -> Dict[str, float]:
        """Calculates retrieval metrics (MRR, Recall@k)."""
        # Similarity matrix: (num_queries, num_codes)
        similarities = self.model.similarity(queries_emb, codes_emb)
        
        # Get ranks: For each query, sort code indices by similarity descending
        # argsort gives the indices of codes.
        sorted_indices = torch.argsort(similarities, descending=True)
        
        # Find where the ground truth (diagonal) is located in the sorted list
        # We assume 1-to-1 mapping where query[i] matches code[i]
        num_samples = len(similarities)
        ranks = torch.zeros(num_samples, dtype=torch.long)
        
        # Efficient vectorization for ranking is complex, utilizing loops for clarity
        # Optimization: In a code search task where Q_i matches C_i, the target index is i.
        # We need to find the position of 'i' in sorted_indices[i]
        
        true_ranks = []
        for i in range(num_samples):
            # Find the rank of the true positive (index i)
            # (sorted_indices[i] == i).nonzero() returns the index in the sorted list
            rank = (sorted_indices[i] == i).nonzero(as_tuple=True)[0].item() + 1
            true_ranks.append(rank)

        return {
            'mrr': np.mean([1.0 / r for r in true_ranks]),
            'recall@1': np.mean([1 if r <= 1 else 0 for r in true_ranks]),
            # 'recall@3': np.mean([1 if r <= 3 else 0 for r in true_ranks]),
            # 'recall@5': np.mean([1 if r <= 5 else 0 for r in true_ranks]),
            'avg_rank': np.mean(true_ranks),
            'num_samples': num_samples
        }

def scan_checkpoints(root_dir: Path) -> Dict[str, Dict[str, Path]]:
    """
    Recursively finds the latest checkpoint for each sub-experiment.
    Returns: { "model_type": { "step_name": Path(latest_ckpt) } }
    """
    if not root_dir.exists():
        logger.error(f"Directory not found: {root_dir}")
        return {}

    structure = {}
    logger.info("Scanning directory structure...")

    # Iterate over model types (e.g., starcoder, qwen)
    for model_type_dir in [d for d in root_dir.iterdir() if d.is_dir()]:
        model_name = model_type_dir.name
        structure[model_name] = {}
        
        # Iterate over steps (e.g., train_step1, train_step2)
        for step_dir in [d for d in model_type_dir.iterdir() if d.is_dir()]:
            # Find all checkpoints
            ckpts = list(step_dir.glob("checkpoint-*"))
            if ckpts:
                # Sort by number in 'checkpoint-XXX'
                latest = max(ckpts, key=lambda p: int(p.name.split('-')[1]))
                structure[model_name][step_dir.name] = latest
                
    return structure

def get_user_selection(options: List[str], prompt_msg: str) -> List[str]:
    """Helper for interactive CLI selection."""
    print(f"\n{prompt_msg}")
    for i, opt in enumerate(options, 1):
        print(f"{i}. {opt}")
    
    while True:
        choice = input("\nEnter numbers (comma-separated) or 'all': ").strip()
        if choice.lower() == 'all':
            return options
        
        try:
            indices = [int(c.strip()) - 1 for c in choice.split(',')]
            selected = [options[i] for i in indices if 0 <= i < len(options)]
            if selected:
                return selected
            print("Selection out of range.")
        except ValueError:
            print("Invalid input format.")

def main():
    # --- Configuration ---
    BASE_DIR = Path("/cpfs01/projects-HDD/cfff-0082a359858b_HDD/lyx_24110240175/000_PseudoBridge")
    BASE_MODELS_DIR = BASE_DIR / "2_Models"
    FINETUNED_DIR = BASE_DIR / "3_Finetune_models/26.01.07_NEW_results"
    DATASETS_DIR = BASE_DIR / "4_Data/000_testdata"
    
    config = EvalConfig(batch_size=64)

    # --- Model Selection ---
    print("="*50 + "\nModel Selection\n" + "="*50)
    mode = input("Select Source:\n1. Base Models\n2. Finetuned Models\nChoice (1/2): ").strip()
    
    selected_paths: List[Path] = []
    
    if mode == "1":
        # Base models
        if not BASE_MODELS_DIR.exists():
            logger.error("Base model directory not found.")
            return
        available = [d.name for d in BASE_MODELS_DIR.iterdir() if d.is_dir()]
        choices = get_user_selection(available, "Available Base Models:")
        selected_paths = [BASE_MODELS_DIR / c for c in choices]
        
    elif mode == "2":
        # Finetuned models
        structure = scan_checkpoints(FINETUNED_DIR)
        if not structure:
            logger.error("No finetuned checkpoints found.")
            return
            
        # Select Model Type
        types = list(structure.keys())
        chosen_types = get_user_selection(types, "Available Model Types:")
        
        # Select Steps for each type
        for m_type in chosen_types:
            steps = list(structure[m_type].keys())
            if not steps: continue
            
            chosen_steps = get_user_selection(steps, f"Select steps for {m_type}:")
            for step in chosen_steps:
                selected_paths.append(structure[m_type][step])
                logger.info(f"Added: {structure[m_type][step]}")
    else:
        logger.error("Invalid choice.")
        return

    # --- Dataset Selection ---
    dataset_files = list(DATASETS_DIR.glob("**/*.jsonl"))
    if not dataset_files:
        logger.error(f"No .jsonl datasets found in {DATASETS_DIR}")
        return
    
    print(f"\nFound {len(dataset_files)} datasets. Evaluating on all...")

    # --- Evaluation Loop ---
    results = []
    
    for model_idx, model_path in enumerate(tqdm(selected_paths, desc="Models"), 1):
        try:
            evaluator = CodeSearchEvaluator(model_path)
            
            for data_path in tqdm(dataset_files, desc="Datasets", leave=False):
                try:
                    queries, codes = evaluator.load_data(data_path, config)
                    if not queries: continue

                    # Compute Embeddings
                    q_embs = evaluator.compute_embeddings(queries, config.batch_size)
                    c_embs = evaluator.compute_embeddings(codes, config.batch_size)
                    
                    # Calculate Metrics
                    metrics = evaluator.evaluate(q_embs, c_embs)
                    
                    # Store Result
                    result_entry = {
                        "model": model_path.name,
                        "parent_dir": model_path.parent.name, # e.g. step name
                        "dataset": data_path.name,
                        "status": "success",
                        **metrics
                    }
                    results.append(result_entry)
                    
                    # Console Output (Simplified)
                    tqdm.write(f"[{model_path.parent.name}] on [{data_path.name}] -> MRR: {metrics['mrr']:.4f}")

                except Exception as e:
                    logger.error(f"Error evaluating {data_path.name}: {e}")
                    results.append({
                        "model": model_path.name, 
                        "dataset": data_path.name, 
                        "status": "failed", 
                        "error": str(e)
                    })
                    
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")

    # --- Save Results ---
    if not results:
        logger.warning("No results to save.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df = pd.DataFrame(results)
    
    # Save JSON
    json_path = f"eval_results_{timestamp}.json"
    df.to_json(json_path, orient="records", indent=2, force_ascii=False)
    
    # Save Excel
    excel_path = f"eval_results_{timestamp}.xlsx"
    df.to_excel(excel_path, index=False)

    logger.info(f"\nEvaluation Complete!")
    logger.info(f"Results saved to: {excel_path}")
    
    # Print Summary of Success
    success_df = df[df["status"] == "success"]
    if not success_df.empty:
        best_run = success_df.loc[success_df["mrr"].idxmax()]
        print("\n" + "="*50)
        print(f"Best Result: MRR {best_run['mrr']:.4f}")
        print(f"Model: {best_run['parent_dir']} / {best_run['model']}")
        print(f"Dataset: {best_run['dataset']}")
        print("="*50)

if __name__ == "__main__":
    main()