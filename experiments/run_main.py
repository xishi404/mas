#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Main experiment runner for MaaS.

End-to-end pipeline:
  Phase 1   - Train controller via Lagrangian-constrained REINFORCE
  Phase 1.5 - Collect DAG traces with the converged controller
  Phase 2   - Train the pruning gate (offline, MSE on Δ_k = R_k - R_N)
  Phase 3   - Test the controller (with the gate) on the held-out test set

When --pruning_gate_path is set, Phase 1.5 and Phase 2 are skipped.
When --test_only is set, Phase 1 is skipped (an existing checkpoint must
be loadable, otherwise --checkpoint_path must be provided).
"""

import argparse
import asyncio
import glob as glob_mod
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from maas.configs.models_config import ModelsConfig
from maas.ext.maas.benchmark.experiment_configs import EXPERIMENT_CONFIGS
from maas.ext.maas.scripts.optimizer import Optimizer
from maas.logs import logger, define_log_level


DATA_PATHS = {
    "MATH": "maas/ext/maas/data/math_train.jsonl",
    "GSM8K": "maas/ext/maas/data/gsm8k_train.jsonl",
    "MMLU_Pro": "maas/ext/maas/data/mmlu_pro_train.jsonl",
}


def _load_pruning_gate(path, device):
    """Load a pre-trained learned pruning gate from a .pth checkpoint."""
    if not path:
        return None
    import torch
    from maas.ext.maas.models.pruning_gate import PruningGate

    gate = PruningGate()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "gate_state_dict" in checkpoint:
        gate.load_state_dict(checkpoint["gate_state_dict"])
        gate._tau = checkpoint.get("tau", 0.1)
    else:
        gate.load_state_dict(checkpoint)
        gate._tau = 0.1
    gate.eval().to(device)
    logger.info(f"Loaded pruning gate from {path} (tau={gate._tau:.4f})")
    return gate


def parse_args():
    parser = argparse.ArgumentParser(description="MaaS main experiment runner")
    parser.add_argument("--dataset", type=str, choices=list(EXPERIMENT_CONFIGS.keys()),
                        required=True)
    parser.add_argument("--sample", type=int, default=4, help="Samples per round")
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--exec_model_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--opt_model_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--latency_weight", type=float, default=0.005)
    parser.add_argument("--cost_weight", type=float, default=3.0)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--max_concurrent", type=int, default=3)
    parser.add_argument("--max_graph_retries", type=int, default=3)
    parser.add_argument("--lagrangian_lr", type=float, default=0.1,
                        help="Learning rate for the dual variable lambda_acc")

    parser.add_argument("--optimized_path", type=str,
                        default="maas/ext/maas/scripts/optimized")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Explicit controller checkpoint path; bypasses auto-detection")
    parser.add_argument("--pruning_gate_path", type=str, default=None,
                        help="Path to a trained pruning gate; if set, Phase 1.5 + 2 are skipped")

    parser.add_argument("--test_only", action="store_true",
                        help="Skip controller training (Phase 1)")
    parser.add_argument("--train_only", action="store_true",
                        help="Skip testing (Phase 3)")
    parser.add_argument("--num_test_runs", type=int, default=1,
                        help="Number of test runs to average")

    parser.add_argument("--gate_epochs", type=int, default=100)
    parser.add_argument("--gate_lr", type=float, default=0.01)

    return parser.parse_args()


def _read_train_results(path):
    train_results_path = os.path.join(path, "train", "results.json")
    if not os.path.exists(train_results_path):
        return None, None
    with open(train_results_path) as f:
        data = json.load(f)
    if not data:
        return None, None
    last = data[-1]
    return last.get("score"), last.get("avg_cost")


def main():
    args = parse_args()

    config = EXPERIMENT_CONFIGS[args.dataset]
    model_tag = args.exec_model_name.replace("-", "_").replace("/", "_").replace(".", "_")
    experiment_name = f"latency_weight_{args.latency_weight:.4f}".replace(".", "_")
    results_path = os.path.join(args.optimized_path, "experiments", model_tag, experiment_name)
    code_path = args.optimized_path

    define_log_level(dataset=args.dataset, weight=str(args.latency_weight),
                     model=model_tag, flags=experiment_name)

    logger.info(f"=== Experiment: {args.dataset} | {experiment_name} ===")
    logger.info(f"Exec model: {args.exec_model_name}, Opt model: {args.opt_model_name}")
    logger.info(f"latency_weight={args.latency_weight}, cost_weight={args.cost_weight}, "
                f"num_layers={args.num_layers}, acc_baseline={config.acc_baseline}")

    models_config = ModelsConfig.default()
    opt_llm_config = models_config.get(args.opt_model_name)
    exec_llm_config = models_config.get(args.exec_model_name)

    optimizer = Optimizer(
        dataset=config.dataset,
        question_type=config.question_type,
        opt_llm_config=opt_llm_config,
        exec_llm_config=exec_llm_config,
        operators=config.operators,
        sample=args.sample,
        round=args.round,
        batch_size=args.batch_size,
        lr=args.lr,
        latency_weight=args.latency_weight,
        cost_weight=args.cost_weight,
        threshold=args.threshold,
        num_layers=args.num_layers,
        max_graph_retries=args.max_graph_retries,
        max_concurrent_tasks=args.max_concurrent,
        acc_baseline=config.acc_baseline,
        lagrangian_lr=args.lagrangian_lr,
        optimized_path=results_path,
        code_path=code_path,
        checkpoint_path=args.checkpoint_path,
        pruning_gate=_load_pruning_gate(args.pruning_gate_path, "cpu"),
    )

    # --- Phase 1: Train controller ---
    if not args.test_only:
        logger.info("PHASE 1: Training controller")
        train_start = time.time()
        try:
            optimizer.optimize("Graph")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        train_time = time.time() - train_start
        logger.info(f"Phase 1 done in {train_time/60:.1f} min")

    # --- Phase 1.5 + 2: Trace collection + Gate training ---
    # Triggered when no pre-trained gate is provided
    if args.pruning_gate_path is None and not args.train_only:
        logger.info("PHASE 1.5: Collecting traces")
        try:
            asyncio.run(optimizer.collect_traces())
        except Exception as e:
            logger.error(f"Trace collection failed: {e}")
            raise

        logger.info("PHASE 2: Training pruning gate")
        trace_dir = os.path.join(optimizer.root_path, "train", f"round_{args.round}")
        gate_output_path = os.path.join(trace_dir, "pruning_gate.pth")
        data_path = DATA_PATHS[args.dataset]

        # Use the trained lambda_acc from the lagrangian curves CSV if present
        curves_files = glob_mod.glob(os.path.join(trace_dir, "*_lagrangian_curves_*.csv"))
        if curves_files:
            latest_curves = max(curves_files, key=os.path.getctime)
            lambda_arg = f"--lagrangian_curves {latest_curves}"
        else:
            lambda_arg = "--lambda_acc 1.0"

        cmd = (
            f"python -m experiments.train_pruning_gate "
            f"--trace_dir {trace_dir} "
            f"--dataset {args.dataset} "
            f"--data_path {data_path} "
            f"--output_path {gate_output_path} "
            f"--latency_weight {args.latency_weight} "
            f"--cost_weight {args.cost_weight} "
            f"--acc_baseline {config.acc_baseline} "
            f"{lambda_arg} "
            f"--epochs {args.gate_epochs} "
            f"--lr {args.gate_lr}"
        )
        logger.info(f"Running: {cmd}")
        if os.system(cmd) != 0:
            raise RuntimeError("Gate training failed")

        gate = _load_pruning_gate(gate_output_path, optimizer.device)
        optimizer.pruning_gate = gate

    # --- Phase 3: Test ---
    if args.train_only:
        logger.info("PHASE 3: Skipped (train_only)")
        return

    logger.info(f"PHASE 3: Testing ({args.num_test_runs} runs)")
    test_scores = []
    for run in range(1, args.num_test_runs + 1):
        logger.info(f"--- Test run {run}/{args.num_test_runs} ---")
        try:
            score = asyncio.run(optimizer.test())
            test_scores.append(score)
            logger.info(f"Run {run} score: {score}")
        except Exception as e:
            logger.error(f"Test run {run} failed: {e}")
            import traceback
            traceback.print_exc()

    if test_scores:
        logger.info(f"Average score: {sum(test_scores) / len(test_scores):.5f}")


if __name__ == "__main__":
    main()
