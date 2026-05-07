import asyncio
import json
import os
import torch
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Tuple
from pydantic import BaseModel, Field
from lamas.actions.action_node import ActionNode
import aiofiles
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
from lamas.configs.models_config import ModelsConfig
from lamas.provider.llm_provider_registry import create_llm_instance
from lamas.logs import logger
from lamas.utils.common import write_json_file
from lamas.ext.lamas.scripts.utils import extract_random_prompt, update_prompt_in_file
from lamas.ext.lamas.scripts.textgrad.textual_gradient import TEXT_GRAD_PROMPT

class TextGrad(BaseModel):
    prompt: str = Field(default="", description="prompt")

class BaseBenchmark(ABC):
    def __init__(
        self,
        name: str,
        file_path: str,
        log_path: str,
        batch_size: int,
        controller: torch.nn.Module,
        operator_embeddings,
        optimizer: torch.optim.Optimizer,
        latency_weight: float = 0.005,
        cost_weight: float = 3.0,
        num_layers: int = 4,
        threshold: float = 0.3,
        max_graph_retries: int = 3,
        max_concurrent_tasks: int = 1,
        acc_baseline: float = 0.0,
        lagrangian_lr: float = 0.1,
        pruning_gate=None,
    ) -> None:
        self.name = name
        self.file_path = file_path
        self.log_path = log_path
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.controller = controller.to(self.device)
        self.operator_embeddings = operator_embeddings.to(self.device)
        self.optimizer = optimizer
        self.latency_weight = latency_weight
        self.cost_weight = cost_weight
        self.num_layers = num_layers
        self.threshold = threshold
        self.max_graph_retries = max_graph_retries
        self.max_concurrent_tasks = max_concurrent_tasks
        self.pruning_gate = pruning_gate

        # All hardcoded to True/default — single experimental configuration
        self.use_latency = True
        self.use_critical_path = True
        self.parallel_execution = True
        self.normalize_rewards = True
        self.use_lagrangian = True
        self.use_scheduler = True
        self.use_tokens = False  # Tokens path removed; latency only

        # Lagrangian constrained optimization: min latency s.t. acc >= acc_baseline
        self.acc_baseline = acc_baseline
        self.lagrangian_lr = lagrangian_lr
        self.cost_limit = 0.0  # Cost constraint disabled (single configuration)
        self.lambda_acc = 1.0
        self.local_model = False
        self.short_prompt = False
        self.use_test_selection = False
        self.cost_baseline = 0.0
        self.cost_tolerance = 0.3
        self.token_weight = 0.00001
        self.virtual_token_rate = 50.0
        self.train_scheduler = False
        self.lambda_cost = 1.0

        # Lagrangian training curves: per-step metrics
        self.lagrangian_curves = {
            "step": [], "acc": [], "latency": [], "cost": [],
            "lambda_acc": [], "acc_violation": [],
            "lambda_cost": [], "cost_violation": [],
        }

        # EMA statistics for reward normalization
        self.reward_ema_mean = None
        self.reward_ema_sq_mean = None
        self.ema_momentum = 0.95

    PASS = "PASS"
    FAIL = "FAIL"

    def log_average_probabilities(self):
        """Log average probabilities for each operator across all problems in each layer"""
        if not hasattr(self, 'prob_aggregator') or not self.prob_aggregator:
            return

        import numpy as np

        logger.info(f"\n{'='*80}")
        logger.info("AVERAGE OPERATOR PROBABILITIES PER LAYER (ACROSS ALL PROBLEMS)")
        logger.info(f"{'='*80}")

        for layer_idx in sorted(self.prob_aggregator.keys()):
            num_problems = len(next(iter(self.prob_aggregator[layer_idx].values())))
            logger.info(f"\nLayer {layer_idx + 1} (averaged over {num_problems} problems):")
            logger.info(f"{'Operator':<25} {'Avg Prob':<12} {'Std Dev':<12} {'Min':<12} {'Max':<12}")
            logger.info("-" * 73)

            # Sort operators by average probability (descending)
            operator_stats = []
            for op_name, probs in self.prob_aggregator[layer_idx].items():
                avg_prob = np.mean(probs)
                std_prob = np.std(probs)
                min_prob = np.min(probs)
                max_prob = np.max(probs)
                operator_stats.append((op_name, avg_prob, std_prob, min_prob, max_prob))

            # Sort by average probability descending
            operator_stats.sort(key=lambda x: x[1], reverse=True)

            for op_name, avg_prob, std_prob, min_prob, max_prob in operator_stats:
                logger.info(
                    f"{op_name:<25} "
                    f"{avg_prob:>6.4f} ({avg_prob*100:>5.2f}%)  "
                    f"{std_prob:>6.4f}     "
                    f"{min_prob:>6.4f}     "
                    f"{max_prob:>6.4f}"
                )

        logger.info(f"{'='*80}\n")

    def log_operator_latency_statistics(self, results=None):
        """Log latency statistics for each operator across all problems

        Args:
            results: List of result tuples containing latency (optional, for accurate total time)
        """
        if not hasattr(self, 'operator_latency_aggregator') or not self.operator_latency_aggregator:
            return

        import numpy as np

        logger.info(f"\n{'='*80}")
        logger.info("OPERATOR LATENCY STATISTICS (ACROSS ALL PROBLEMS)")
        logger.info(f"{'='*80}")

        # Calculate total latency across all operators (this counts parallel operators multiple times)
        total_operator_latency_naive = sum(sum(latencies) for latencies in self.operator_latency_aggregator.values())

        # Get end-to-end problem time from results (same as used for avg/P90 latency calculation)
        total_problem_time = None
        if results is not None and len(results) > 0:
            # Extract latency from results (index 6: latency is 7th element in tuple)
            # Results format: (input_text, prediction, expected_output, score, cost, logprob, latency)
            total_problem_time = sum(r[6] for r in results)
            logger.info(f"Using end-to-end latency from results: {total_problem_time:.2f}s total")

        # Fallback to bottleneck aggregator if results not provided
        if total_problem_time is None and hasattr(self, 'bottleneck_aggregator') and self.bottleneck_aggregator.get('total_time'):
            total_problem_time = sum(self.bottleneck_aggregator['total_time'])
            logger.info(f"Using latency from bottleneck aggregator: {total_problem_time:.2f}s total (graph-only)")

        # Prepare statistics for each operator
        operator_stats = []
        for op_name in sorted(self.operator_latency_aggregator.keys()):
            latencies = self.operator_latency_aggregator[op_name]
            iterations = self.operator_iteration_aggregator.get(op_name, [1] * len(latencies))

            count = len(latencies)
            total_op_latency = sum(latencies)
            avg_latency = np.mean(latencies)
            std_latency = np.std(latencies)
            min_latency = np.min(latencies)
            max_latency = np.max(latencies)

            # Proportion of operator execution time (naive sum - counts parallel ops multiple times)
            proportion_of_operators = (total_op_latency / total_operator_latency_naive * 100) if total_operator_latency_naive > 0 else 0.0

            # Proportion of TOTAL problem time (uses end-to-end latency you recorded)
            proportion_of_total = (total_op_latency / total_problem_time * 100) if total_problem_time and total_problem_time > 0 else None

            # Calculate average iterations
            avg_iterations = np.mean(iterations) if iterations else 1.0
            total_iterations = sum(iterations)

            operator_stats.append({
                'name': op_name,
                'count': count,
                'total_latency': total_op_latency,
                'avg_latency': avg_latency,
                'std_latency': std_latency,
                'min_latency': min_latency,
                'max_latency': max_latency,
                'proportion_of_operators': proportion_of_operators,
                'proportion_of_total': proportion_of_total,
                'avg_iterations': avg_iterations,
                'total_iterations': total_iterations
            })

        # Sort by proportion of total time (or operator time if total not available)
        sort_key = 'proportion_of_total' if operator_stats[0]['proportion_of_total'] is not None else 'proportion_of_operators'
        operator_stats.sort(key=lambda x: x[sort_key] if x[sort_key] is not None else 0, reverse=True)

        # Display table with both proportions
        if total_problem_time:
            logger.info(f"\n{'Operator':<18} {'Count':<8} {'Total(s)':<10} {'Avg(s)':<10} {'% Ops':<10} {'% TOTAL':<10} {'Avg Iter':<10}")
            logger.info("-" * 84)
        else:
            logger.info(f"\n{'Operator':<18} {'Count':<8} {'Total(s)':<10} {'Avg(s)':<10} {'% Ops':<10} {'Avg Iter':<10}")
            logger.info("-" * 74)

        for stats in operator_stats:
            if total_problem_time:
                logger.info(
                    f"{stats['name']:<18} "
                    f"{stats['count']:<8} "
                    f"{stats['total_latency']:<10.2f} "
                    f"{stats['avg_latency']:<10.2f} "
                    f"{stats['proportion_of_operators']:>6.2f}%   "
                    f"{stats['proportion_of_total']:>6.2f}%   "
                    f"{stats['avg_iterations']:<10.2f}"
                )
            else:
                logger.info(
                    f"{stats['name']:<18} "
                    f"{stats['count']:<8} "
                    f"{stats['total_latency']:<10.2f} "
                    f"{stats['avg_latency']:<10.2f} "
                    f"{stats['proportion_of_operators']:>6.2f}%   "
                    f"{stats['avg_iterations']:<10.2f}"
                )

        if total_problem_time:
            logger.info("-" * 84)
            # Calculate what % of total time is accounted for by summed operator times
            # Note: This may be >100% due to parallel operators being counted multiple times
            total_pct_of_total = (total_operator_latency_naive / total_problem_time * 100) if total_problem_time > 0 else 0
            logger.info(f"{'TOTAL OPERATORS':<18} {'':<8} {total_operator_latency_naive:<10.2f} {'':<10} {'100.00%':<10} {total_pct_of_total:>6.2f}%")
            if total_pct_of_total > 100:
                logger.info(f"{'NOTE':<18} {'':<8} {'Parallel operators counted multiple times in Total(s) column':<60}")
        else:
            logger.info("-" * 74)
            logger.info(f"{'TOTAL OPERATORS':<18} {'':<8} {total_operator_latency_naive:<10.2f} {'':<10} {'100.00%':<10}")

        # Add bottleneck identification
        logger.info(f"\n{'='*80}")
        logger.info("OPERATOR BOTTLENECK IDENTIFICATION")
        logger.info(f"{'='*80}\n")

        # Identify top bottlenecks
        top_3 = operator_stats[:3]

        for i, stats in enumerate(top_3, 1):
            pct = stats['proportion_of_total'] if stats['proportion_of_total'] is not None else stats['proportion_of_operators']
            pct_label = "of TOTAL time" if stats['proportion_of_total'] is not None else "of operator time"

            if i == 1:
                marker = "🔴 PRIMARY BOTTLENECK"
            elif i == 2:
                marker = "🟡 SECONDARY BOTTLENECK"
            else:
                marker = "🟢 TERTIARY"

            logger.info(f"{marker}: {stats['name']}")
            logger.info(f"   {pct:.2f}% {pct_label} | Avg: {stats['avg_latency']:.2f}s | Count: {stats['count']} | Avg Iterations: {stats['avg_iterations']:.2f}")

            # Add operator-specific recommendations
            self._log_operator_optimization_tips(stats['name'], stats)
            logger.info("")

        logger.info(f"{'='*80}\n")

    def _log_operator_optimization_tips(self, operator_name, stats):
        """Provide operator-specific optimization recommendations"""
        tips = {
            'Generate': [
                "→ Consider using a faster/cheaper model for initial generation",
                "→ Implement response caching for similar problems",
                "→ Reduce max_tokens if responses are longer than needed"
            ],
            'GenerateCoT': [
                "→ CoT adds reasoning steps (slower but more accurate)",
                "→ Consider switching to 'Generate' if accuracy is acceptable",
                "→ Use streaming to reduce perceived latency"
            ],
            'MultiGenerateCoT': [
                f"→ Generating {stats['avg_iterations']:.1f} solutions on average",
                "→ Reduce number of generated solutions if acceptable",
                "→ These run sequentially - major latency contributor"
            ],
            'Test': [
                f"→ Averaging {stats['avg_iterations']:.1f} refinement iterations",
                "→ High iterations = many test failures requiring LLM refinement",
                "→ Improve initial generation quality to reduce test-fix loops",
                "→ Consider timeout limits on refinement loops"
            ],
            'SelfRefine': [
                "→ Refines existing solution (LLM call)",
                "→ Consider if refinement actually improves results",
                "→ Could combine with Test operator to avoid redundant calls"
            ],
            'ScEnsemble': [
                "→ Selects best from multiple solutions (LLM call)",
                "→ Only useful if you have multiple diverse solutions",
                "→ Consider rule-based selection if possible"
            ],
            'EarlyStop': [
                "→ Should have minimal latency (control flow only)",
                "→ If showing latency, check implementation"
            ]
        }

        operator_tips = tips.get(operator_name, ["→ No specific optimization tips available"])
        for tip in operator_tips:
            logger.info(f"   {tip}")

    def log_bottleneck_analysis(self):
        """Log framework bottleneck analysis showing where time is spent"""
        if not hasattr(self, 'bottleneck_aggregator') or not self.bottleneck_aggregator['total_time']:
            return

        import numpy as np

        logger.info(f"\n{'='*80}")
        logger.info("FRAMEWORK BOTTLENECK ANALYSIS")
        logger.info(f"{'='*80}")

        # Calculate statistics
        controller_times = self.bottleneck_aggregator['controller_time']
        operator_times = self.bottleneck_aggregator['operator_time']
        overhead_times = self.bottleneck_aggregator['overhead_time']
        total_times = self.bottleneck_aggregator['total_time']

        num_problems = len(total_times)

        # Calculate averages
        avg_controller = np.mean(controller_times)
        avg_operator = np.mean(operator_times)
        avg_overhead = np.mean(overhead_times)
        avg_total = np.mean(total_times)

        # Calculate proportions (as percentages)
        prop_controller = (avg_controller / avg_total * 100) if avg_total > 0 else 0
        prop_operator = (avg_operator / avg_total * 100) if avg_total > 0 else 0
        prop_overhead = (avg_overhead / avg_total * 100) if avg_total > 0 else 0

        # Calculate standard deviations
        std_controller = np.std(controller_times)
        std_operator = np.std(operator_times)
        std_overhead = np.std(overhead_times)

        logger.info(f"\nAnalyzed {num_problems} problems")
        logger.info(f"Average total time per problem: {avg_total:.2f}s\n")

        logger.info(f"{'Component':<25} {'Avg Time (s)':<15} {'Std Dev (s)':<15} {'Proportion':<15}")
        logger.info("-" * 70)

        logger.info(f"{'Controller (selection)':<25} {avg_controller:<15.4f} {std_controller:<15.4f} {prop_controller:>6.2f}%")
        logger.info(f"{'Operator Execution':<25} {avg_operator:<15.4f} {std_operator:<15.4f} {prop_operator:>6.2f}%")
        logger.info(f"{'Framework Overhead':<25} {avg_overhead:<15.4f} {std_overhead:<15.4f} {prop_overhead:>6.2f}%")
        logger.info("-" * 70)
        logger.info(f"{'TOTAL':<25} {avg_total:<15.4f} {'':<15} {'100.00%':<15}")

        logger.info(f"\n{'='*80}")
        logger.info("BOTTLENECK INTERPRETATION")
        logger.info(f"{'='*80}")

        # Provide interpretation
        if prop_operator > 70:
            logger.info("🔴 BOTTLENECK: Operator Execution (>70% of time)")
            logger.info("   → Focus on optimizing operators (LLM calls, code execution)")
            logger.info("   → Consider parallelization or caching strategies")
        elif prop_controller > 30:
            logger.info("🟡 NOTABLE: Controller overhead is significant (>30%)")
            logger.info("   → Consider optimizing controller architecture")
            logger.info("   → Check if embedding computation can be cached")
        elif prop_overhead > 20:
            logger.info("🟡 NOTABLE: Framework overhead is significant (>20%)")
            logger.info("   → Check asyncio coordination efficiency")
            logger.info("   → Review data processing and logging overhead")
        else:
            logger.info("🟢 BALANCED: No single dominant bottleneck")
            logger.info("   → System is reasonably well-balanced")

        logger.info(f"{'='*80}\n")

    async def load_data(self, specific_indices: List[int] = None) -> List[dict]:
        data = []
        async with aiofiles.open(self.file_path, mode="r", encoding="utf-8") as file:
            async for line in file:
                data.append(json.loads(line))
        if specific_indices is not None:
            filtered_data = [data[i] for i in specific_indices if i < len(data)]
            return filtered_data
        return data

    def save_results_to_csv(self, results: List[Tuple[Any, ...]], columns: List[str]):
        df = pd.DataFrame(results, columns=columns)
        avg_score = df["score"].mean()
        avg_cost = df["cost"].mean() if "cost" in df.columns else 0.0
        avg_latency = df["latency"].mean() if "latency" in df.columns else 0.0
        p90_latency = df["latency"].quantile(0.9) if "latency" in df.columns else 0.0
        avg_cp_token = df["cp_token"].mean() if "cp_token" in df.columns else 0.0
        # Keep cost column in CSV (user wants total cost preserved)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Include penalty type in CSV filename
        penalty_suffix = ""
        if hasattr(self, 'use_tokens') and self.use_tokens:
            # Token-based penalty mode (weight is divided by 50 in loss calculation)
            penalty_suffix = f"_tok{self.token_weight:.4f}".replace(".", "_")
        elif hasattr(self, 'latency_weight') and hasattr(self, 'use_latency'):
            # Latency-based penalty mode
            if self.use_latency:
                penalty_suffix = f"_lat{self.latency_weight:.4f}".replace(".", "_")
            else:
                penalty_suffix = "_no_penalty"

        # Include parallel execution mode in CSV filename
        parallel_suffix = ""
        if hasattr(self, 'parallel_execution'):
            parallel_suffix = "_parallel" if self.parallel_execution else "_sequential"

        # Include critical path mode in CSV filename
        critical_path_suffix = ""
        if hasattr(self, 'use_critical_path') and self.use_critical_path:
            critical_path_suffix = "_cp"

        # Include normalization mode in CSV filename
        norm_suffix = ""
        if hasattr(self, 'normalize_rewards') and self.normalize_rewards:
            norm_suffix = "_norm"

        # Include threshold in CSV filename
        threshold_suffix = ""
        if hasattr(self, 'threshold'):
            threshold_suffix = f"_th{self.threshold:.2f}".replace(".", "_")

        # Include cost weight in CSV filename
        cost_weight_suffix = ""
        if hasattr(self, 'cost_weight'):
            cost_weight_suffix = f"_cw{self.cost_weight:.1f}".replace(".", "_")

        # Include num_layers in CSV filename
        num_layers_suffix = ""
        if hasattr(self, 'num_layers'):
            num_layers_suffix = f"_nl{self.num_layers}"

        # Include short_prompt in CSV filename
        short_prompt_suffix = ""
        if hasattr(self, 'short_prompt') and self.short_prompt:
            short_prompt_suffix = "_sp"

        # Include lagrangian mode and lr in CSV filename
        lagrangian_suffix = ""
        if hasattr(self, 'use_lagrangian') and self.use_lagrangian:
            lagrangian_suffix = f"_lag{self.lagrangian_lr:.3f}".replace(".", "_")

        # Include scheduler mode in CSV filename
        scheduler_suffix = "_scheduler" if (getattr(self, 'use_scheduler', False) or getattr(self, 'train_scheduler', False)) else ""
        trsched_suffix = "_trsched" if getattr(self, 'train_scheduler', False) else ""
        testsel_suffix = "_testsel" if getattr(self, 'use_test_selection', False) else ""
        pruning_suffix = "_pruning" if getattr(self, "pruning_gate", None) is not None else ""
        small_suffix = "_small" if getattr(self, 'small', False) else ""

        filename = f"{avg_score:.5f}_{current_time}{penalty_suffix}{parallel_suffix}{critical_path_suffix}{norm_suffix}{threshold_suffix}{cost_weight_suffix}{num_layers_suffix}{short_prompt_suffix}{lagrangian_suffix}{scheduler_suffix}{trsched_suffix}{testsel_suffix}{pruning_suffix}{small_suffix}.csv"
        output_file = os.path.join(self.log_path, filename)
        df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
        return avg_score, avg_cost, avg_latency, p90_latency, avg_cp_token

    def save_operator_detail_csv(self):
        """Save per-operator (output_tokens, wall_clock_time) detail CSV for local_model mode.
        Also computes CP_len and CP_wallclock per problem offline."""
        if not hasattr(self, '_operator_detail_records') or not self._operator_detail_records:
            return

        detail_df = pd.DataFrame(self._operator_detail_records)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        detail_filename = f"operator_detail_{current_time}.csv"
        detail_path = os.path.join(self.log_path, detail_filename)
        detail_df.to_csv(detail_path, index=False)
        logger.info(f"Per-operator detail CSV saved to {detail_path}")

        # Compute CP_len and CP_wallclock per problem offline
        cp_records = []
        for problem_idx in detail_df['problem_idx'].unique():
            prob_df = detail_df[detail_df['problem_idx'] == problem_idx]
            cp_len = 0.0
            cp_wallclock = 0.0
            for layer_idx in prob_df['layer'].unique():
                layer_df = prob_df[prob_df['layer'] == layer_idx]
                cp_len += layer_df['output_tokens'].max()
                cp_wallclock += layer_df['wall_clock_time'].max()
            cp_records.append({
                'problem_idx': problem_idx,
                'cp_len': cp_len,
                'cp_wallclock': cp_wallclock,
            })
        cp_df = pd.DataFrame(cp_records)
        cp_filename = f"cp_metrics_{current_time}.csv"
        cp_path = os.path.join(self.log_path, cp_filename)
        cp_df.to_csv(cp_path, index=False)
        logger.info(f"CP metrics CSV saved to {cp_path}")
        logger.info(f"  Avg CP_len: {cp_df['cp_len'].mean():.2f}, Avg CP_wallclock: {cp_df['cp_wallclock'].mean():.4f}s")
        logger.info(f"  Correlation(CP_len, CP_wallclock): {cp_df['cp_len'].corr(cp_df['cp_wallclock']):.4f}")

    def log_mismatch(
        self,
        problem: str,
        expected_output: Any,
        prediction: str,
        extracted_output: Any,
        extract_answer_code: str = "None",
    ):
        log_data = {
            "question": problem,
            "right_answer": expected_output,
            "model_output": prediction,
            "extracted_output": extracted_output,
            "extract_answer_code": extract_answer_code,
        }
        log_file = Path(self.log_path) / "log.json"
        if log_file.exists():
            with log_file.open("r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []
        data.append(log_data)
        write_json_file(log_file, data, encoding="utf-8", indent=4)

    def _compute_critical_path_loss(
        self,
        batch_layer_operator_infos: list,
        scores_tensor: torch.Tensor,  # [batch_size]
        costs_tensor: torch.Tensor,   # [batch_size]
        latencies_tensor: torch.Tensor  # [batch_size]
    ) -> tuple:
        """
        Compute loss with soft critical-path-aware credit assignment.

        Each operator's latency penalty is weighted by its longest-path-through ratio:
            weight_i = longest_path_through(i) / T
        where longest_path_through(i) = earliest_finish(i) + longest_chain_after(i)
        and T is the problem's end-to-end latency. Operators on the actual critical
        path get weight = 1.0; off-CP operators get weight = 1 - slack(i)/T.

        Falls back to the per-layer max heuristic when path lengths are unavailable.

        Args:
            batch_layer_operator_infos: List of dicts with layer operator information.
                Expected keys: log_probs_per_layer, operator_latencies_per_layer,
                and (optionally) operator_path_lengths_per_layer.
            scores_tensor: Per-problem scores [batch_size] - GLOBAL
            costs_tensor: Per-problem costs [batch_size] - GLOBAL
            latencies_tensor: Per-problem total latencies [batch_size] - GLOBAL
        """
        all_log_probs = []
        all_utilities = []

        for problem_idx in range(len(batch_layer_operator_infos)):
            layer_info = batch_layer_operator_infos[problem_idx]
            problem_score = scores_tensor[problem_idx].item()
            problem_cost = costs_tensor[problem_idx].item()
            problem_latency = latencies_tensor[problem_idx].item()

            log_probs_per_layer = layer_info['log_probs_per_layer']
            operator_latencies_per_layer = layer_info['operator_latencies_per_layer']
            operator_path_lengths_per_layer = layer_info.get('operator_path_lengths_per_layer')

            for layer_idx, (layer_log_probs, layer_latencies) in enumerate(
                zip(log_probs_per_layer, operator_latencies_per_layer)
            ):
                if len(layer_latencies) == 0:
                    continue

                if operator_path_lengths_per_layer is not None and problem_latency > 0:
                    layer_path_lengths = operator_path_lengths_per_layer[layer_idx]
                    weights = [pl / problem_latency for pl in layer_path_lengths]
                else:
                    max_latency = max(layer_latencies)
                    weights = [1.0 if lat == max_latency else 0.0 for lat in layer_latencies]

                for op_idx in range(len(layer_latencies)):
                    utility = (problem_score
                               - self.cost_weight * problem_cost
                               - weights[op_idx] * self.latency_weight * problem_latency)
                    all_log_probs.append(layer_log_probs[op_idx])
                    all_utilities.append(utility)

        if len(all_log_probs) > 0:
            log_probs_tensor = torch.stack(all_log_probs)
            utilities_tensor = torch.tensor(all_utilities, dtype=torch.float32, device=self.device)

            # Store raw average utility before normalization for logging
            avg_utility = utilities_tensor.mean().item()

            # Apply EMA-based reward normalization if enabled
            if self.normalize_rewards:
                import math
                batch_mean = utilities_tensor.mean().item()
                batch_sq_mean = (utilities_tensor ** 2).mean().item()
                if self.reward_ema_mean is None:
                    self.reward_ema_mean = batch_mean
                    self.reward_ema_sq_mean = batch_sq_mean
                else:
                    self.reward_ema_mean = self.ema_momentum * self.reward_ema_mean + (1 - self.ema_momentum) * batch_mean
                    self.reward_ema_sq_mean = self.ema_momentum * self.reward_ema_sq_mean + (1 - self.ema_momentum) * batch_sq_mean
                ema_var = max(self.reward_ema_sq_mean - self.reward_ema_mean ** 2, 1e-8)
                utilities_tensor = (utilities_tensor - self.reward_ema_mean) / math.sqrt(ema_var)

            loss = -(log_probs_tensor * utilities_tensor).mean()
            return loss, avg_utility
        else:
            return torch.tensor(0.0, device=self.device, requires_grad=True), 0.0

    def _compute_critical_path_loss_tokens(
        self,
        batch_layer_operator_infos: list,
        scores_tensor: torch.Tensor,
        costs_tensor: torch.Tensor
    ) -> tuple:
        """
        Critical path loss using ACTUAL TOKENS from critical path operators only.

        For each layer:
        - Identify max token operator (critical path) using cp_token_count
        - Only critical path operator gets penalized
        - Penalty uses sum of critical path tokens across all layers (not global token sum)
        """
        all_log_probs = []
        all_utilities = []

        for problem_idx in range(len(batch_layer_operator_infos)):
            layer_info = batch_layer_operator_infos[problem_idx]
            problem_score = scores_tensor[problem_idx].item()
            problem_cost = costs_tensor[problem_idx].item()

            log_probs_per_layer = layer_info['log_probs_per_layer']
            operator_token_counts_per_layer = layer_info['operator_token_counts_per_layer']

            # Calculate problem_tokens by summing only critical path tokens
            problem_tokens = 0.0
            critical_path_indices = []  # Track which operators are on critical path

            for layer_tokens in operator_token_counts_per_layer:
                if len(layer_tokens) == 0:
                    critical_path_indices.append(None)
                    continue
                # Find critical path operator (max tokens in this layer)
                max_tokens = max(layer_tokens)
                max_token_idx = layer_tokens.index(max_tokens)
                critical_path_indices.append(max_token_idx)
                problem_tokens += max_tokens

            # Assign utilities to each operator
            for layer_idx, (layer_log_probs, layer_tokens) in enumerate(zip(
                log_probs_per_layer, operator_token_counts_per_layer
            )):
                if len(layer_tokens) == 0:
                    continue

                max_token_idx = critical_path_indices[layer_idx]

                for op_idx in range(len(layer_tokens)):
                    if op_idx == max_token_idx:
                        # Critical path operator gets penalized by total critical path tokens
                        # Divide token_weight by 50 to align scale with latency weight
                        utility = problem_score - self.cost_weight * problem_cost - (self.token_weight / 50.0) * problem_tokens
                    else:
                        # Non-critical operators don't get penalty
                        utility = problem_score - self.cost_weight * problem_cost

                    all_log_probs.append(layer_log_probs[op_idx])
                    all_utilities.append(utility)

        # Apply normalization and compute loss (same as latency version)
        if len(all_log_probs) > 0:
            log_probs_tensor = torch.stack(all_log_probs)
            utilities_tensor = torch.tensor(all_utilities, dtype=torch.float32, device=self.device)

            # Store raw average utility before normalization for logging
            avg_utility = utilities_tensor.mean().item()

            # Apply EMA-based reward normalization if enabled
            if self.normalize_rewards:
                import math
                batch_mean = utilities_tensor.mean().item()
                batch_sq_mean = (utilities_tensor ** 2).mean().item()
                if self.reward_ema_mean is None:
                    self.reward_ema_mean = batch_mean
                    self.reward_ema_sq_mean = batch_sq_mean
                else:
                    self.reward_ema_mean = self.ema_momentum * self.reward_ema_mean + (1 - self.ema_momentum) * batch_mean
                    self.reward_ema_sq_mean = self.ema_momentum * self.reward_ema_sq_mean + (1 - self.ema_momentum) * batch_sq_mean
                ema_var = max(self.reward_ema_sq_mean - self.reward_ema_mean ** 2, 1e-8)
                utilities_tensor = (utilities_tensor - self.reward_ema_mean) / math.sqrt(ema_var)

            loss = -(log_probs_tensor * utilities_tensor).mean()
            return loss, avg_utility
        else:
            return torch.tensor(0.0, device=self.device, requires_grad=True), 0.0

    def _compute_lagrangian_loss(
        self,
        per_problem_logprobs,
        scores_tensor: torch.Tensor,
        costs_tensor: torch.Tensor,
        latencies_tensor: torch.Tensor,
        output_tokens_tensor: torch.Tensor,
        virtual_tokens_tensor: torch.Tensor,
    ) -> tuple:
        """
        Lagrangian constrained optimization with baseline-normalized constraints:
            min  E[latency]  (or token penalty)
            s.t. E[acc] >= acc_baseline
                 E[cost] <= cost_limit   (if cost_baseline > 0)

        Normalization: divide score by acc_baseline, cost by cost_limit so both
        constraint terms are dimensionless (~1 at baseline). This makes the dual
        variables comparable in scale and converge at similar rates.

        Policy gradient on θ:
            utility_i = λ_acc * (score_i / acc_baseline) - penalty_i - λ_cost * (cost_i / cost_limit)
            loss = -(logprobs * utilities).mean()

        Dual gradient on λ (normalized violations):
            λ_acc  += lr * (1 - avg_acc / acc_baseline)
            λ_cost += lr * (avg_cost / cost_limit - 1)
        """
        lambda_acc = self.lambda_acc
        lambda_cost = self.lambda_cost if self.cost_limit > 0 else self.cost_weight

        if per_problem_logprobs and isinstance(per_problem_logprobs[0], torch.Tensor):
            logprobs = torch.stack(per_problem_logprobs).to(self.device)
        else:
            logprobs = torch.tensor(per_problem_logprobs, dtype=torch.float32, device=self.device)

        # Compute latency/token penalty per problem
        if self.use_tokens:
            if self.use_critical_path and self.parallel_execution:
                penalty = (self.token_weight / 50.0) * output_tokens_tensor
            else:
                penalty = (self.token_weight / 50.0) * virtual_tokens_tensor
        elif self.use_latency:
            penalty = self.latency_weight * latencies_tensor
        else:
            penalty = torch.zeros_like(latencies_tensor)

        # Lagrangian utility with baseline normalization
        norm_scores = scores_tensor / self.acc_baseline if self.acc_baseline > 0 else scores_tensor
        if self.cost_limit > 0:
            norm_costs = costs_tensor / self.cost_limit
            utilities = lambda_acc * norm_scores - penalty - lambda_cost * norm_costs
        else:
            utilities = lambda_acc * norm_scores - penalty - self.cost_weight * costs_tensor

        avg_utility = utilities.mean().item()

        # Apply reward normalization if enabled (EMA second-moment method)
        if self.normalize_rewards:
            import math
            batch_mean = utilities.mean().item()
            batch_sq_mean = (utilities ** 2).mean().item()
            if self.reward_ema_mean is None:
                self.reward_ema_mean = batch_mean
                self.reward_ema_sq_mean = batch_sq_mean
            else:
                self.reward_ema_mean = self.ema_momentum * self.reward_ema_mean + (1 - self.ema_momentum) * batch_mean
                self.reward_ema_sq_mean = self.ema_momentum * self.reward_ema_sq_mean + (1 - self.ema_momentum) * batch_sq_mean
            ema_var = max(self.reward_ema_sq_mean - self.reward_ema_mean ** 2, 1e-8)
            utilities = (utilities - self.reward_ema_mean) / math.sqrt(ema_var)

        loss = -(logprobs * utilities).mean()

        # Dual variable updates with normalized violations (dimensionless)
        batch_avg_acc = scores_tensor.mean().item()
        acc_violation = 1.0 - batch_avg_acc / self.acc_baseline if self.acc_baseline > 0 else 0.0  # >0 means violated
        batch_avg_cost = costs_tensor.mean().item()
        cost_violation = batch_avg_cost / self.cost_limit - 1.0 if self.cost_limit > 0 else 0.0  # >0 means violated

        # Projected gradient ascent: λ = max(0, λ + lr * violation)
        self.lambda_acc = max(0.0, self.lambda_acc + self.lagrangian_lr * acc_violation)
        if self.cost_limit > 0:
            self.lambda_cost = max(0.0, self.lambda_cost + self.lagrangian_lr * cost_violation)

        return loss, avg_utility, lambda_acc, batch_avg_acc, acc_violation, lambda_cost, batch_avg_cost, cost_violation

    def _compute_lagrangian_critical_path_loss(
        self,
        batch_layer_operator_infos: list,
        scores_tensor: torch.Tensor,
        costs_tensor: torch.Tensor,
        latencies_tensor: torch.Tensor,
    ) -> tuple:
        """
        Lagrangian loss with critical path aware credit assignment.
        Only critical path operators receive the latency penalty.
        Score and cost are baseline-normalized for balanced dual variable dynamics.
        """
        lambda_acc = self.lambda_acc
        lambda_cost = self.lambda_cost if self.cost_limit > 0 else self.cost_weight

        all_log_probs = []
        all_utilities = []

        for problem_idx in range(len(batch_layer_operator_infos)):
            layer_info = batch_layer_operator_infos[problem_idx]
            problem_score = scores_tensor[problem_idx].item()
            problem_cost = costs_tensor[problem_idx].item()
            problem_latency = latencies_tensor[problem_idx].item()

            # Baseline normalization
            norm_score = problem_score / self.acc_baseline if self.acc_baseline > 0 else problem_score
            if self.cost_limit > 0:
                norm_cost = problem_cost / self.cost_limit
            else:
                norm_cost = problem_cost  # fallback: no normalization, use raw cost

            log_probs_per_layer = layer_info['log_probs_per_layer']
            operator_names_per_layer = layer_info['operator_names_per_layer']
            operator_latencies_per_layer = layer_info['operator_latencies_per_layer']

            for layer_log_probs, layer_names, layer_latencies in zip(
                log_probs_per_layer, operator_names_per_layer, operator_latencies_per_layer
            ):
                if len(layer_latencies) == 0:
                    continue

                max_latency = max(layer_latencies)
                max_latency_idx = layer_latencies.index(max_latency)

                cost_term = lambda_cost * norm_cost if self.cost_limit > 0 else self.cost_weight * norm_cost

                for op_idx in range(len(layer_latencies)):
                    if op_idx == max_latency_idx:
                        utility = lambda_acc * norm_score - cost_term - self.latency_weight * problem_latency
                    else:
                        utility = lambda_acc * norm_score - cost_term

                    all_log_probs.append(layer_log_probs[op_idx])
                    all_utilities.append(utility)

        if len(all_log_probs) > 0:
            log_probs_tensor = torch.stack(all_log_probs)
            utilities_tensor = torch.tensor(all_utilities, dtype=torch.float32, device=self.device)
            avg_utility = utilities_tensor.mean().item()

            if self.normalize_rewards:
                import math
                batch_mean = utilities_tensor.mean().item()
                batch_sq_mean = (utilities_tensor ** 2).mean().item()
                if self.reward_ema_mean is None:
                    self.reward_ema_mean = batch_mean
                    self.reward_ema_sq_mean = batch_sq_mean
                else:
                    self.reward_ema_mean = self.ema_momentum * self.reward_ema_mean + (1 - self.ema_momentum) * batch_mean
                    self.reward_ema_sq_mean = self.ema_momentum * self.reward_ema_sq_mean + (1 - self.ema_momentum) * batch_sq_mean
                ema_var = max(self.reward_ema_sq_mean - self.reward_ema_mean ** 2, 1e-8)
                utilities_tensor = (utilities_tensor - self.reward_ema_mean) / math.sqrt(ema_var)

            loss = -(log_probs_tensor * utilities_tensor).mean()
        else:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            avg_utility = 0.0

        # Dual variable updates with normalized violations
        batch_avg_acc = scores_tensor.mean().item()
        acc_violation = 1.0 - batch_avg_acc / self.acc_baseline if self.acc_baseline > 0 else 0.0
        batch_avg_cost = costs_tensor.mean().item()
        cost_violation = batch_avg_cost / self.cost_limit - 1.0 if self.cost_limit > 0 else 0.0
        # Projected gradient ascent: λ = max(0, λ + lr * violation)
        self.lambda_acc = max(0.0, self.lambda_acc + self.lagrangian_lr * acc_violation)
        if self.cost_limit > 0:
            self.lambda_cost = max(0.0, self.lambda_cost + self.lagrangian_lr * cost_violation)

        return loss, avg_utility, lambda_acc, batch_avg_acc, acc_violation, lambda_cost, batch_avg_cost, cost_violation

    def _compute_lagrangian_critical_path_loss_tokens(
        self,
        batch_layer_operator_infos: list,
        scores_tensor: torch.Tensor,
        costs_tensor: torch.Tensor,
    ) -> tuple:
        """
        Lagrangian loss with critical path aware credit assignment using tokens.
        Score and cost are baseline-normalized for balanced dual variable dynamics.
        """
        lambda_acc = self.lambda_acc
        lambda_cost = self.lambda_cost if self.cost_limit > 0 else self.cost_weight

        all_log_probs = []
        all_utilities = []

        for problem_idx in range(len(batch_layer_operator_infos)):
            layer_info = batch_layer_operator_infos[problem_idx]
            problem_score = scores_tensor[problem_idx].item()
            problem_cost = costs_tensor[problem_idx].item()

            # Baseline normalization
            norm_score = problem_score / self.acc_baseline if self.acc_baseline > 0 else problem_score
            if self.cost_limit > 0:
                norm_cost = problem_cost / self.cost_limit
            else:
                norm_cost = problem_cost

            log_probs_per_layer = layer_info['log_probs_per_layer']
            operator_token_counts_per_layer = layer_info['operator_token_counts_per_layer']

            problem_tokens = 0.0
            critical_path_indices = []
            for layer_tokens in operator_token_counts_per_layer:
                if len(layer_tokens) == 0:
                    critical_path_indices.append(None)
                    continue
                max_tokens = max(layer_tokens)
                max_token_idx = layer_tokens.index(max_tokens)
                critical_path_indices.append(max_token_idx)
                problem_tokens += max_tokens

            cost_term = lambda_cost * norm_cost if self.cost_limit > 0 else self.cost_weight * norm_cost

            for layer_idx, (layer_log_probs, layer_tokens) in enumerate(zip(
                log_probs_per_layer, operator_token_counts_per_layer
            )):
                if len(layer_tokens) == 0:
                    continue
                max_token_idx = critical_path_indices[layer_idx]
                for op_idx in range(len(layer_tokens)):
                    if op_idx == max_token_idx:
                        utility = lambda_acc * norm_score - cost_term - (self.token_weight / 50.0) * problem_tokens
                    else:
                        utility = lambda_acc * norm_score - cost_term

                    all_log_probs.append(layer_log_probs[op_idx])
                    all_utilities.append(utility)

        if len(all_log_probs) > 0:
            log_probs_tensor = torch.stack(all_log_probs)
            utilities_tensor = torch.tensor(all_utilities, dtype=torch.float32, device=self.device)
            avg_utility = utilities_tensor.mean().item()

            if self.normalize_rewards:
                import math
                batch_mean = utilities_tensor.mean().item()
                batch_sq_mean = (utilities_tensor ** 2).mean().item()
                if self.reward_ema_mean is None:
                    self.reward_ema_mean = batch_mean
                    self.reward_ema_sq_mean = batch_sq_mean
                else:
                    self.reward_ema_mean = self.ema_momentum * self.reward_ema_mean + (1 - self.ema_momentum) * batch_mean
                    self.reward_ema_sq_mean = self.ema_momentum * self.reward_ema_sq_mean + (1 - self.ema_momentum) * batch_sq_mean
                ema_var = max(self.reward_ema_sq_mean - self.reward_ema_mean ** 2, 1e-8)
                utilities_tensor = (utilities_tensor - self.reward_ema_mean) / math.sqrt(ema_var)

            loss = -(log_probs_tensor * utilities_tensor).mean()
        else:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            avg_utility = 0.0

        # Dual variable updates with normalized violations
        batch_avg_acc = scores_tensor.mean().item()
        acc_violation = 1.0 - batch_avg_acc / self.acc_baseline if self.acc_baseline > 0 else 0.0
        batch_avg_cost = costs_tensor.mean().item()
        cost_violation = batch_avg_cost / self.cost_limit - 1.0 if self.cost_limit > 0 else 0.0
        # Projected gradient ascent: λ = max(0, λ + lr * violation)
        self.lambda_acc = max(0.0, self.lambda_acc + self.lagrangian_lr * acc_violation)
        if self.cost_limit > 0:
            self.lambda_cost = max(0.0, self.lambda_cost + self.lagrangian_lr * cost_violation)

        return loss, avg_utility, lambda_acc, batch_avg_acc, acc_violation, lambda_cost, batch_avg_cost, cost_violation

    @abstractmethod
    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple[Any, ...]:
        pass

    @abstractmethod
    def calculate_score(self, expected_output: Any, prediction: Any) -> Tuple[float, Any]:
        pass

    @abstractmethod
    def get_result_columns(self) -> List[str]:
        pass

    async def evaluate_all_problems(self, data: List[dict], graph: Callable, max_concurrent_tasks: int = 1, repetitions: int = 4, is_textgrad: bool = False):
        semaphore = asyncio.Semaphore(max_concurrent_tasks)
        results = []
        textgrad = False
        prev_rep_score = None

        # Track total tokens for training
        self.total_training_cost = 0.0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        async def sem_evaluate(problem):
            async with semaphore:
                try:
                    return await self.evaluate_problem(problem, graph)
                except Exception as e:
                    logger.error(f"Error evaluating problem: {e}")
                    return ("", "", "", 0.0, 0.0, 0.0, 0.0)
        
        for rep in range(1, repetitions + 1):
            logger.info(f"Starting training repetition {rep}/{repetitions}")
            rep_scores = []
            cp_batches_count = 0  # Track how many batches used critical path loss
            total_batches_count = 0

            if textgrad and is_textgrad:
                prompt_name, prompt_content = extract_random_prompt(self.log_path)
                textgrad_prompt = TEXT_GRAD_PROMPT.format(dataset = self.name, prompt_name = prompt_name, prompt_content = prompt_content)
                textgrad_llm_config = ModelsConfig.default().get("gpt-4o-mini")
                textgrad_llm = create_llm_instance(textgrad_llm_config)
                textgrad_node = await ActionNode.from_pydantic(TextGrad).fill(context=textgrad_prompt, mode="xml_fill", llm=textgrad_llm)
                response = textgrad_node.instruct_content.model_dump()
                update_prompt_in_file(prompt_name, response["prompt"])
                is_textgrad = False

            for batch_start in range(0, len(data), self.batch_size):
                batch = data[batch_start:batch_start + self.batch_size]

                # Track output tokens before batch for per-problem token tracking
                tokens_before_batch = graph.llm.cost_manager.get_total_completion_tokens() if hasattr(graph, 'llm') and hasattr(graph.llm, 'cost_manager') else 0

                tasks = [sem_evaluate(problem) for problem in batch]
                batch_results = await tqdm_asyncio.gather(
                    *tasks,
                    desc=f"Repetition {rep}: Executing batch {batch_start // self.batch_size + 1}",
                    total=len(batch)
                )
                results.extend(batch_results)

                # Extract data from results, including layer operator info for critical path tracking
                batch_layer_operator_infos = []
                per_problem_logprobs = []  # For backward compatibility
                scores = []
                costs = []
                cp_tokens = []  # NEW: Track cp_token per problem
                latencies = []
                output_tokens_list = []  # Track output tokens per problem

                for r in batch_results:
                    score = float(r[3]) if r[3] is not None else 0.0
                    cost = float(r[4]) if r[4] is not None else 0.0
                    logprob = r[5]
                    cp_token = float(r[6]) if r[6] is not None else 0.0  # NEW: Extract cp_token (now at index 6)
                    latency = float(r[7]) if r[7] is not None else 0.0  # MOVED: latency now at index 7
                    layer_operator_info = r[8] if len(r) > 8 else None  # MOVED: layer_operator_info now at index 8

                    # Calculate output tokens for this problem from layer_operator_info
                    if layer_operator_info is not None and 'operator_token_counts_per_layer' in layer_operator_info:
                        # Sum all tokens across all layers and operators for this problem
                        problem_tokens = sum(
                            sum(layer_tokens) for layer_tokens in layer_operator_info['operator_token_counts_per_layer']
                        )
                    else:
                        # Fallback: estimate from batch total (will be less accurate)
                        problem_tokens = 0.0

                    per_problem_logprobs.append(logprob)
                    scores.append(score)
                    costs.append(cost)  # cost is now per-problem (computed from tokens in graph)
                    cp_tokens.append(cp_token)  # NEW: Track cp_token per problem
                    latencies.append(latency)
                    output_tokens_list.append(problem_tokens)
                    batch_layer_operator_infos.append(layer_operator_info)
                    rep_scores.append(score)

                # Accumulate tokens from the graph's cost manager after each batch
                if hasattr(graph, 'llm') and hasattr(graph.llm, 'cost_manager'):
                    cost_manager = graph.llm.cost_manager
                    self.total_prompt_tokens = cost_manager.get_total_prompt_tokens()
                    self.total_completion_tokens = cost_manager.get_total_completion_tokens()
                    self.total_training_cost = cost_manager.get_total_cost()  # Track cumulative cost

                if len(per_problem_logprobs) > 0:
                    scores_tensor = torch.tensor(scores, dtype=torch.float32, device=self.device)
                    costs_tensor = torch.tensor(costs, dtype=torch.float32, device=self.device)
                    latencies_tensor = torch.tensor(latencies, dtype=torch.float32, device=self.device)
                    output_tokens_tensor = torch.tensor(output_tokens_list, dtype=torch.float32, device=self.device)

                    # Compute virtual tokens: output_tokens + (latency * virtual_token_rate)
                    virtual_tokens_tensor = output_tokens_tensor + (latencies_tensor * self.virtual_token_rate)

                    # Choose loss calculation based on flags
                    # Lagrangian mode takes precedence when enabled
                    lagrangian_info = None  # Will hold (lambda_acc, batch_avg_acc, acc_violation, lambda_cost, batch_avg_cost, cost_violation) if lagrangian
                    if self.use_lagrangian:
                        # Lagrangian constrained optimization
                        if self.use_tokens and self.use_critical_path and self.parallel_execution and all(info is not None for info in batch_layer_operator_infos):
                            loss, avg_utility, lag_lambda, lag_acc, lag_violation, lag_lambda_cost, lag_cost, lag_cost_violation = self._compute_lagrangian_critical_path_loss_tokens(
                                batch_layer_operator_infos, scores_tensor, costs_tensor
                            )
                        elif self.use_latency and self.use_critical_path and self.parallel_execution and all(info is not None for info in batch_layer_operator_infos):
                            loss, avg_utility, lag_lambda, lag_acc, lag_violation, lag_lambda_cost, lag_cost, lag_cost_violation = self._compute_lagrangian_critical_path_loss(
                                batch_layer_operator_infos, scores_tensor, costs_tensor, latencies_tensor
                            )
                        else:
                            loss, avg_utility, lag_lambda, lag_acc, lag_violation, lag_lambda_cost, lag_cost, lag_cost_violation = self._compute_lagrangian_loss(
                                per_problem_logprobs, scores_tensor, costs_tensor,
                                latencies_tensor, output_tokens_tensor, virtual_tokens_tensor
                            )
                        lagrangian_info = (lag_lambda, lag_acc, lag_violation, lag_lambda_cost, lag_cost, lag_cost_violation)
                        # Record training curves
                        step = len(self.lagrangian_curves['step'])
                        self.lagrangian_curves['step'].append(step)
                        self.lagrangian_curves['acc'].append(lag_acc)
                        self.lagrangian_curves['latency'].append(latencies_tensor.mean().item())
                        self.lagrangian_curves['cost'].append(lag_cost)
                        self.lagrangian_curves['lambda_acc'].append(lag_lambda)
                        self.lagrangian_curves['acc_violation'].append(lag_violation)
                        self.lagrangian_curves['lambda_cost'].append(lag_lambda_cost)
                        self.lagrangian_curves['cost_violation'].append(lag_cost_violation)
                    elif self.use_tokens and self.use_critical_path and self.parallel_execution and all(info is not None for info in batch_layer_operator_infos):
                        # Critical path with actual tokens from critical path operators only (parallel execution only)
                        loss, avg_utility = self._compute_critical_path_loss_tokens(
                            batch_layer_operator_infos,
                            scores_tensor,
                            costs_tensor
                        )
                    elif self.use_tokens:
                        # Standard virtual token penalty (no critical path)
                        # Divide token_weight by 50 to align scale with latency weight
                        if isinstance(per_problem_logprobs[0], torch.Tensor):
                            logprobs = torch.stack(per_problem_logprobs).to(self.device)
                        else:
                            logprobs = torch.tensor(per_problem_logprobs, dtype=torch.float32, device=self.device)
                        utilities = scores_tensor - self.cost_weight * costs_tensor - (self.token_weight / 50.0) * virtual_tokens_tensor

                        # Store raw average utility before normalization for logging
                        avg_utility = utilities.mean().item()

                        # Apply reward normalization if enabled
                        if self.normalize_rewards:
                            std = utilities.std(unbiased=False)
                            if std > 1e-8:
                                utilities = (utilities - utilities.mean()) / std
                            else:
                                utilities = utilities - utilities.mean()

                        loss = -(logprobs * utilities).mean()
                    elif self.use_latency and self.use_critical_path and self.parallel_execution and all(info is not None for info in batch_layer_operator_infos):
                        # Critical path aware credit assignment with latency (parallel execution only)
                        loss, avg_utility = self._compute_critical_path_loss(
                            batch_layer_operator_infos,
                            scores_tensor,
                            costs_tensor,
                            latencies_tensor
                        )
                    elif self.use_latency:
                        # Original behavior: all operators get same utility
                        if isinstance(per_problem_logprobs[0], torch.Tensor):
                            logprobs = torch.stack(per_problem_logprobs).to(self.device)
                        else:
                            logprobs = torch.tensor(per_problem_logprobs, dtype=torch.float32, device=self.device)
                        utilities = scores_tensor - self.cost_weight * costs_tensor - self.latency_weight * latencies_tensor

                        # Store raw average utility before normalization for logging
                        avg_utility = utilities.mean().item()

                        # Apply reward normalization if enabled
                        if self.normalize_rewards:
                            std = utilities.std(unbiased=False)  # Use unbiased=False to avoid NaN with single element
                            if std > 1e-8:  # Only normalize if there's actual variance
                                utilities = (utilities - utilities.mean()) / std
                            else:
                                utilities = utilities - utilities.mean()  # Just center if no variance

                        loss = -(logprobs * utilities).mean()
                    else:
                        # No latency consideration
                        logprobs = torch.stack(per_problem_logprobs).to(self.device)
                        utilities = scores_tensor - self.cost_weight * costs_tensor

                        # Store raw average utility before normalization for logging
                        avg_utility = utilities.mean().item()

                        # Apply reward normalization if enabled
                        if self.normalize_rewards:
                            std = utilities.std(unbiased=False)  # Use unbiased=False to avoid NaN with single element
                            if std > 1e-8:  # Only normalize if there's actual variance
                                utilities = (utilities - utilities.mean()) / std
                            else:
                                utilities = utilities - utilities.mean()  # Just center if no variance

                        loss = -(logprobs * utilities).mean()

                    avg_batch_cost = costs_tensor.mean().item()
                    avg_batch_latency = latencies_tensor.mean().item()
                    avg_batch_score = scores_tensor.mean().item()
                    avg_batch_virtual_tokens = virtual_tokens_tensor.mean().item()

                    # Calculate average critical path tokens if in critical path mode
                    # Use the cp_token values that were extracted from graph results (includes post-processing)
                    if self.use_tokens and self.use_critical_path and self.parallel_execution:
                        cp_tokens_tensor = torch.tensor(cp_tokens, dtype=torch.float32, device=self.device)
                        avg_batch_cp_tokens = cp_tokens_tensor.mean().item()
                    else:
                        avg_batch_cp_tokens = avg_batch_virtual_tokens

                    # Note: avg_utility is already computed in each branch above

                    # Get sum of logprobs (already computed above)
                    if isinstance(per_problem_logprobs[0], torch.Tensor):
                        logprobs_tensor = torch.stack(per_problem_logprobs)
                    else:
                        logprobs_tensor = torch.tensor(per_problem_logprobs, device=self.device)
                    sum_logprobs = logprobs_tensor.sum().item()

                    if loss.requires_grad:
                        loss.backward()
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        batch_idx = batch_start // self.batch_size + 1
                        if lagrangian_info is not None:
                            lag_lambda, lag_acc, lag_violation, lag_lambda_cost, lag_cost, lag_cost_violation = lagrangian_info
                            penalty_str = f"CP_Tokens: {avg_batch_cp_tokens:.1f}" if self.use_tokens else f"Latency: {avg_batch_latency:.2f}s"
                            cost_constraint_str = f", λ_cost: {lag_lambda_cost:.4f}, CostViolation: {lag_cost_violation:.6f}" if self.cost_limit > 0 else ""
                            logger.info(f"Repetition {rep}: Batch {batch_idx} Loss: {loss.item():.4f}, Score: {avg_batch_score:.4f}, Cost: {avg_batch_cost:.6f}, {penalty_str}, Utility: {avg_utility:.4f}, λ_acc: {lag_lambda:.4f}, AccViolation: {lag_violation:.4f}{cost_constraint_str}, SumLogProbs: {sum_logprobs:.4f}")
                        elif self.use_tokens:
                            if self.use_critical_path and self.parallel_execution:
                                logger.info(f"Repetition {rep}: Batch {batch_idx} Loss: {loss.item():.4f}, Score: {avg_batch_score:.4f}, Cost: {avg_batch_cost:.6f}, CP_Tokens: {avg_batch_cp_tokens:.1f}, Utility: {avg_utility:.4f}, SumLogProbs: {sum_logprobs:.4f}")
                            else:
                                logger.info(f"Repetition {rep}: Batch {batch_idx} Loss: {loss.item():.4f}, Score: {avg_batch_score:.4f}, Cost: {avg_batch_cost:.6f}, VirtualTokens: {avg_batch_virtual_tokens:.1f}, Utility: {avg_utility:.4f}, SumLogProbs: {sum_logprobs:.4f}")
                        else:
                            logger.info(f"Repetition {rep}: Batch {batch_idx} Loss: {loss.item():.4f}, Score: {avg_batch_score:.4f}, Cost: {avg_batch_cost:.6f}, Latency: {avg_batch_latency:.2f}s, Utility: {avg_utility:.4f}, SumLogProbs: {sum_logprobs:.4f}")
                    else:
                        batch_idx = batch_start // self.batch_size + 1
                        if lagrangian_info is not None:
                            lag_lambda, lag_acc, lag_violation, lag_lambda_cost, lag_cost, lag_cost_violation = lagrangian_info
                            penalty_str = f"CP_Tokens: {avg_batch_cp_tokens:.1f}" if self.use_tokens else f"Latency: {avg_batch_latency:.2f}s"
                            cost_constraint_str = f", λ_cost: {lag_lambda_cost:.4f}, CostViolation: {lag_cost_violation:.6f}" if self.cost_limit > 0 else ""
                            logger.info(f"Repetition {rep}: Batch {batch_idx} Loss does not require grad and was skipped. Score: {avg_batch_score:.4f}, Cost: {avg_batch_cost:.6f}, {penalty_str}, Utility: {avg_utility:.4f}, λ_acc: {lag_lambda:.4f}, AccViolation: {lag_violation:.4f}{cost_constraint_str}, SumLogProbs: {sum_logprobs:.4f}")
                        elif self.use_tokens:
                            if self.use_critical_path and self.parallel_execution:
                                logger.info(f"Repetition {rep}: Batch {batch_idx} Loss does not require grad and was skipped. Score: {avg_batch_score:.4f}, Cost: {avg_batch_cost:.6f}, CP_Tokens: {avg_batch_cp_tokens:.1f}, Utility: {avg_utility:.4f}, SumLogProbs: {sum_logprobs:.4f}")
                            else:
                                logger.info(f"Repetition {rep}: Batch {batch_idx} Loss does not require grad and was skipped. Score: {avg_batch_score:.4f}, Cost: {avg_batch_cost:.6f}, VirtualTokens: {avg_batch_virtual_tokens:.1f}, Utility: {avg_utility:.4f}, SumLogProbs: {sum_logprobs:.4f}")
                        else:
                            logger.info(f"Repetition {rep}: Batch {batch_idx} Loss does not require grad and was skipped. Score: {avg_batch_score:.4f}, Cost: {avg_batch_cost:.6f}, Latency: {avg_batch_latency:.2f}s, Utility: {avg_utility:.4f}, SumLogProbs: {sum_logprobs:.4f}")
                else:
                    logger.info(f"Repetition {rep}: Batch {batch_start // self.batch_size + 1} skipped due to invalid logprobs.")

            if rep_scores:
                current_rep_score = sum(rep_scores) / len(rep_scores)
            else:
                current_rep_score = 0.0

            if not textgrad:
                if prev_rep_score is not None and current_rep_score < prev_rep_score:
                    textgrad = True
                prev_rep_score = current_rep_score

        return results
    
    async def evaluate_all_problems_test(self, data: List[dict], graph: Callable, max_concurrent_tasks: int = 1):
        # local_model mode: force sequential query execution for clean latency measurement
        if self.local_model:
            max_concurrent_tasks = 1
        semaphore = asyncio.Semaphore(max_concurrent_tasks)

        # Initialize probability aggregation structures
        self.prob_aggregator = {}  # {layer_idx: {operator_name: [probabilities]}}

        # Initialize operator latency aggregation structures
        self.operator_latency_aggregator = {}  # {operator_name: [latencies]}
        self.operator_iteration_aggregator = {}  # {operator_name: [iteration_counts]}

        # Initialize bottleneck aggregation structures
        self.bottleneck_aggregator = {
            'controller_time': [],
            'operator_time': [],
            'overhead_time': [],
            'total_time': []
        }

        async def sem_evaluate(problem):
            async with semaphore:
                result = await self.evaluate_problem(problem, graph)

                # Collect probabilities from the graph if available
                if hasattr(graph, 'last_probs_layers'):
                    for layer_idx, probs in enumerate(graph.last_probs_layers):
                        if layer_idx not in self.prob_aggregator:
                            self.prob_aggregator[layer_idx] = {}

                        # Get operator names from graph
                        if hasattr(graph, 'selection_operator_names'):
                            probs_cpu = probs.detach().cpu().numpy()
                            for op_idx, op_name in enumerate(graph.selection_operator_names):
                                if op_name not in self.prob_aggregator[layer_idx]:
                                    self.prob_aggregator[layer_idx][op_name] = []
                                self.prob_aggregator[layer_idx][op_name].append(float(probs_cpu[op_idx]))

                # Collect operator latencies from the graph if available
                if hasattr(graph, 'last_operator_latencies'):
                    for op_name, latencies in graph.last_operator_latencies.items():
                        if op_name not in self.operator_latency_aggregator:
                            self.operator_latency_aggregator[op_name] = []
                        self.operator_latency_aggregator[op_name].extend(latencies)

                # Collect operator iterations from the graph if available
                if hasattr(graph, 'last_operator_iterations'):
                    for op_name, iterations in graph.last_operator_iterations.items():
                        if op_name not in self.operator_iteration_aggregator:
                            self.operator_iteration_aggregator[op_name] = []
                        self.operator_iteration_aggregator[op_name].extend(iterations)

                # Collect bottleneck breakdown from the graph if available
                if hasattr(graph, 'last_bottleneck_breakdown'):
                    breakdown = graph.last_bottleneck_breakdown
                    self.bottleneck_aggregator['controller_time'].append(breakdown['controller_time'])
                    self.bottleneck_aggregator['operator_time'].append(breakdown['operator_time'])
                    self.bottleneck_aggregator['overhead_time'].append(breakdown['overhead_time'])
                    self.bottleneck_aggregator['total_time'].append(breakdown['total_time'])

                return result

        tasks = [sem_evaluate(problem) for problem in data]
        results = await tqdm_asyncio.gather(*tasks, desc=f"Evaluating {self.name} problems", total=len(data))

        # Convert results to format expected by CSV (keep cost column)
        # CSV format: (input_text, prediction, expected_output, score, cost, logprob, cp_token, latency) - 8 columns
        processed_results = []
        # local_model mode: also collect per-operator detail for separate CSV
        if self.local_model:
            self._operator_detail_records = []
        for r_idx, r in enumerate(results):
            # r format: (input_text, prediction, expected_output, score, cost, logprob, cp_token, latency, layer_operator_info)
            if len(r) >= 9:
                # New format with cp_token and layer_operator_info
                # Drop only layer_operator_info (r[8]), keep cost (r[4])
                processed_result = (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7])
                # local_model mode: extract per-operator detail from layer_operator_info
                if self.local_model and r[8] is not None:
                    layer_op_info = r[8]
                    op_names_per_layer = layer_op_info.get('operator_names_per_layer', [])
                    op_tokens_per_layer = layer_op_info.get('operator_token_counts_per_layer', [])
                    op_latencies_per_layer = layer_op_info.get('operator_latencies_per_layer', [])
                    for layer_idx in range(len(op_names_per_layer)):
                        names = op_names_per_layer[layer_idx]
                        tokens = op_tokens_per_layer[layer_idx] if layer_idx < len(op_tokens_per_layer) else []
                        latencies = op_latencies_per_layer[layer_idx] if layer_idx < len(op_latencies_per_layer) else []
                        for op_i, op_name in enumerate(names):
                            self._operator_detail_records.append({
                                'problem_idx': r_idx,
                                'layer': layer_idx,
                                'operator': op_name,
                                'output_tokens': tokens[op_i] if op_i < len(tokens) else 0,
                                'wall_clock_time': latencies[op_i] if op_i < len(latencies) else 0.0,
                            })
            elif len(r) >= 8:
                # New format with cp_token but no layer_operator_info
                # Keep all 8 values including cost
                processed_result = (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7])
            else:
                # Old format (backward compatibility) - add cp_token as 0.0
                # r format: (input, pred, expected, score, cost, logprob, latency)
                processed_result = (r[0], r[1], r[2], r[3], r[4], r[5], 0.0, r[6])
            processed_results.append(processed_result)

        return processed_results
    
    async def run_evaluation(self, graph: Callable, va_list: List[int], is_test: bool, sample: int, is_textgrad: bool = False):
        data = await self.load_data(va_list)

        if is_test == True:
            results = await self.evaluate_all_problems_test(data, graph, self.max_concurrent_tasks)
            columns = self.get_result_columns()
            average_score, average_cost, average_latency, p90_latency, average_cp_token = self.save_results_to_csv(results, columns)
            if self.local_model:
                cost_label = "cost(tokens)"
            else:
                cost_label = "cost($)"
            logger.info(f"Average score on {self.name} dataset: {average_score:.5f}, Average {cost_label}: {average_cost:.6f}, Average latency: {average_latency:.2f}s, P90 latency: {p90_latency:.2f}s, Avg cp_token: {average_cp_token:.2f}")

            # local_model mode: save per-operator detail CSV
            if self.local_model:
                self.save_operator_detail_csv()

            # Track token usage and cost for testing
            if hasattr(graph, 'llm') and hasattr(graph.llm, 'cost_manager'):
                cost_manager = graph.llm.cost_manager
                self.test_prompt_tokens = cost_manager.get_total_prompt_tokens()
                self.test_completion_tokens = cost_manager.get_total_completion_tokens()
                self.test_total_tokens = self.test_prompt_tokens + self.test_completion_tokens
                self.test_total_cost = cost_manager.get_total_cost()
                logger.info(f"Test token usage - Prompt: {self.test_prompt_tokens:,}, Completion: {self.test_completion_tokens:,}, Total: {self.test_total_tokens:,}")
                if self.local_model:
                    logger.info(f"Test total cost (tokens): {self.test_total_cost:,.0f}")
                else:
                    logger.info(f"Test total cost: ${self.test_total_cost:.6f}")

            # Log average probabilities across the dataset
            self.log_average_probabilities()

            # Log operator latency statistics (pass results for accurate end-to-end latency)
            self.log_operator_latency_statistics(results=results)

            # Log framework bottleneck analysis
            self.log_bottleneck_analysis()

            return average_score

        results = await self.evaluate_all_problems(data, graph, self.max_concurrent_tasks, sample, is_textgrad)

        columns = self.get_result_columns()
        # Filter out layer_operator_info (9th element, index 8) before saving to CSV
        # Keep first 8 elements: problem, prediction, expected_output, score, cost, logprob, cp_token, latency
        results_for_csv = [result[:8] if len(result) == 9 else result for result in results]
        average_score, average_cost, average_latency, p90_latency, average_cp_token = self.save_results_to_csv(results_for_csv, columns)
        if self.local_model:
            cost_label = "cost(tokens)"
        else:
            cost_label = "cost($)"
        logger.info(f"Average score on {self.name} dataset: {average_score:.5f}, Average {cost_label}: {average_cost:.6f}, Average latency: {average_latency:.2f}s, P90 latency: {p90_latency:.2f}s, Avg cp_token: {average_cp_token:.2f}")

        try:
            os.makedirs(self.log_path, exist_ok=True)
            # Include penalty type in checkpoint filename
            penalty_suffix = ""
            if hasattr(self, 'use_tokens') and self.use_tokens:
                # Token-based penalty mode (weight is divided by 50 in loss calculation)
                penalty_suffix = f"_tok{self.token_weight:.4f}".replace(".", "_")
            elif hasattr(self, 'latency_weight') and hasattr(self, 'use_latency'):
                # Latency-based penalty mode
                if self.use_latency:
                    penalty_suffix = f"_lat{self.latency_weight:.4f}".replace(".", "_")
                else:
                    penalty_suffix = "_no_penalty"
            # Include parallel execution mode in checkpoint filename
            parallel_suffix = ""
            if hasattr(self, 'parallel_execution'):
                parallel_suffix = "_parallel" if self.parallel_execution else "_sequential"
            # Include critical path mode in checkpoint filename
            critical_path_suffix = ""
            if hasattr(self, 'use_critical_path') and self.use_critical_path:
                critical_path_suffix = "_cp"
            # Include normalization mode in checkpoint filename
            norm_suffix = ""
            if hasattr(self, 'normalize_rewards') and self.normalize_rewards:
                norm_suffix = "_norm"
            # Include threshold in checkpoint filename
            threshold_suffix = ""
            if hasattr(self, 'threshold'):
                threshold_suffix = f"_th{self.threshold:.2f}".replace(".", "_")
            # Include cost weight in checkpoint filename
            cost_weight_suffix = ""
            if hasattr(self, 'cost_weight'):
                cost_weight_suffix = f"_cw{self.cost_weight:.1f}".replace(".", "_")
            # Include num_layers in checkpoint filename
            num_layers_suffix = ""
            if hasattr(self, 'num_layers'):
                num_layers_suffix = f"_nl{self.num_layers}"
            # Include short_prompt in checkpoint filename
            short_prompt_suffix = ""
            if hasattr(self, 'short_prompt') and self.short_prompt:
                short_prompt_suffix = "_sp"
            # Include lagrangian mode and lr in checkpoint filename
            lagrangian_suffix = ""
            if hasattr(self, 'use_lagrangian') and self.use_lagrangian:
                lagrangian_suffix = f"_lag{self.lagrangian_lr:.3f}".replace(".", "_")
            # Include scheduler mode in checkpoint filename
            scheduler_suffix = "_scheduler" if (getattr(self, 'use_scheduler', False) or getattr(self, 'train_scheduler', False)) else ""
            trsched_suffix = "_trsched" if getattr(self, 'train_scheduler', False) else ""
            testsel_suffix = "_testsel" if getattr(self, 'use_test_selection', False) else ""
            pruning_suffix = "_pruning" if getattr(self, "pruning_gate", None) is not None else ""
            small_suffix = "_small" if getattr(self, 'small', False) else ""
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            controller_path = os.path.join(self.log_path, f"{self.name}_controller_sample{sample}{penalty_suffix}{parallel_suffix}{critical_path_suffix}{norm_suffix}{threshold_suffix}{cost_weight_suffix}{num_layers_suffix}{short_prompt_suffix}{lagrangian_suffix}{scheduler_suffix}{trsched_suffix}{testsel_suffix}{pruning_suffix}{small_suffix}_{timestamp}.pth")
            torch.save(self.controller.state_dict(), controller_path)
            logger.info(f"Saved controller parameters to {controller_path}")

            # Save Lagrangian training curves if available
            if self.use_lagrangian and len(self.lagrangian_curves['step']) > 0:
                import csv
                curves_path = os.path.join(self.log_path, f"{self.name}_lagrangian_curves_sample{sample}{penalty_suffix}{parallel_suffix}{critical_path_suffix}{cost_weight_suffix}{num_layers_suffix}{short_prompt_suffix}{lagrangian_suffix}{scheduler_suffix}{trsched_suffix}{testsel_suffix}{pruning_suffix}{small_suffix}_{timestamp}.csv")
                with open(curves_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['step', 'acc', 'latency', 'cost', 'lambda_acc', 'acc_violation', 'lambda_cost', 'cost_violation'])
                    for i in range(len(self.lagrangian_curves['step'])):
                        writer.writerow([
                            self.lagrangian_curves['step'][i],
                            self.lagrangian_curves['acc'][i],
                            self.lagrangian_curves['latency'][i],
                            self.lagrangian_curves['cost'][i],
                            self.lagrangian_curves['lambda_acc'][i],
                            self.lagrangian_curves['acc_violation'][i],
                            self.lagrangian_curves['lambda_cost'][i],
                            self.lagrangian_curves['cost_violation'][i],
                        ])
                logger.info(f"Saved Lagrangian training curves ({len(self.lagrangian_curves['step'])} steps) to {curves_path}")

            logger.info("Successfully Finish Training")
        except Exception as e:
            logger.error(f"Failed to save controller parameters: {e}")

        return average_score
