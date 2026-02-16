"""Data processing for SGTR-RL training.

This module handles loading cached generations from self-rec-framework
and creating DPO training triples.
"""

from .eval_loader import load_eval_file, load_experiment_evals, EvalSample, categorize_sample
from .triple_generator import create_dpo_triples, DPOTriple

__all__ = [
    "load_eval_file",
    "load_experiment_evals",
    "EvalSample",
    "categorize_sample",
    "create_dpo_triples",
    "DPOTriple",
]
