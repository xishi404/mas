"""Phase 2 of the gate pipeline: train the pruning gate from execution traces.

Inputs:
  - DAG operator traces (JSONL) collected during Phase 1.5
  - The dataset's training file (for ground-truth answers)

Procedure:
  1. Load all traces under --trace_dir, group by problem execution
  2. For each stop point k in the DAG, compute:
       S_k = correctness via majority vote + ground truth
       T_k = wall-clock latency to k
       C_k = cumulative cost to k
       R_k = lambda_acc * (S_k / acc_baseline) - lambda_t * T_k - lambda_c * C_k
       delta_k = R_k - R_N    (advantage of stopping at k vs running everything)
  3. Train a small MLP gate (state features + query embedding) with MSE loss on delta_k
  4. Compute tau = std({delta_k | delta_k > 0}) for the inference sigmoid

Usage:
    python -m experiments.train_pruning_gate \\
        --trace_dir <path/to/traces/round_1> \\
        --dataset GSM8K \\
        --data_path lamas/ext/lamas/data/gsm8k_train.jsonl \\
        --output_path pruning_gate.pth \\
        --latency_weight 0.005 --cost_weight 3.0 \\
        --acc_baseline 0.85 --lambda_acc 1.0
"""

import argparse
import csv
import json
import os
import re
import string
import sys
from datetime import datetime
from math import isclose
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lamas.ext.lamas.models.pruning_gate import PruningGate
from lamas.ext.lamas.models.utils import get_sentence_embedding
from lamas.ext.lamas.scripts.optimized.dag_executor import _consensus_key


# ============================================================
# Dataset-specific evaluators
# ============================================================

class DatasetEvaluator:
    """Base class for dataset-specific answer extraction and correctness."""

    def extract_answer(self, solution_text: str):
        """Extract the answer from a full solution text. Returns None if extraction fails."""
        raise NotImplementedError

    def check_correct(self, predicted, expected) -> bool:
        """Check if predicted answer matches expected answer."""
        raise NotImplementedError

    def evaluate_at_k(self, terminal_solutions: List[str], ground_truth) -> float:
        """Compute correctness (0 or 1) from solutions collected up to stop point k.

        Default: majority vote on extracted answers, then compare to ground truth.
        """
        answers = []
        for sol in terminal_solutions:
            ans = self.extract_answer(sol)
            if ans is not None:
                answers.append(ans)
        if not answers:
            return 0.0
        majority = self._majority_vote(answers)
        return 1.0 if self.check_correct(majority, ground_truth) else 0.0

    def _majority_vote(self, answers):
        """Select majority answer using equivalence-aware grouping."""
        groups = []  # [(representative, count)]
        for ans in answers:
            matched = False
            for i, (rep, count) in enumerate(groups):
                if self.check_correct(ans, rep):
                    groups[i] = (rep, count + 1)
                    matched = True
                    break
            if not matched:
                groups.append((ans, 1))
        groups.sort(key=lambda x: x[1], reverse=True)
        return groups[0][0]


class MATHEvaluator(DatasetEvaluator):
    """MATH dataset: extract \\boxed{...}, compare with SymPy."""

    def extract_answer(self, text: str):
        if not text:
            return None
        pattern = r"\\boxed{((?:[^{}]|{[^{}]*})*)}"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[-1].strip()
        # Fallback: last sentence
        sentences = re.split(r"(?<!\d)[.!?]\s+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences[-1] if sentences else None

    def check_correct(self, predicted, expected) -> bool:
        if predicted is None or expected is None:
            return False
        if str(predicted) == str(expected):
            return True
        try:
            if self._is_digit(predicted) and self._is_digit(expected):
                return isclose(self._parse_digit(predicted), self._parse_digit(expected), abs_tol=1e-3)
        except Exception:
            pass
        try:
            return self._symbolic_equal(predicted, expected)
        except Exception:
            pass
        return False

    def _is_digit(self, num):
        return self._parse_digit(num) is not None

    def _parse_digit(self, num):
        import regex
        num = regex.sub(",", "", str(num))
        try:
            return float(num)
        except Exception:
            if str(num).endswith("%"):
                num = str(num)[:-1].rstrip("\\")
                try:
                    return float(num) / 100
                except Exception:
                    pass
        return None

    def _symbolic_equal(self, a, b):
        from sympy import N, simplify
        from sympy.parsing.latex import parse_latex
        from sympy.parsing.sympy_parser import parse_expr

        def _parse(s):
            for f in [parse_latex, parse_expr]:
                try:
                    return f(s)
                except Exception:
                    pass
            return s

        pa, pb = _parse(a), _parse(b)
        try:
            if simplify(pa - pb) == 0:
                return True
        except Exception:
            pass
        try:
            if isclose(float(N(pa)), float(N(pb)), abs_tol=1e-3):
                return True
        except Exception:
            pass
        return False

    def get_ground_truth(self, problem: dict):
        """Extract ground truth answer from dataset problem."""
        return self.extract_answer(problem["solution"])


class GSM8KEvaluator(DatasetEvaluator):
    """GSM8K dataset: extract last number, compare numerically."""

    def extract_answer(self, text: str):
        if not text:
            return None
        matches = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?|\d+\.\d+", str(text))
        if matches:
            try:
                return float(matches[-1].replace(",", ""))
            except ValueError:
                return None
        return None

    def check_correct(self, predicted, expected) -> bool:
        if predicted is None or expected is None:
            return False
        try:
            return abs(float(predicted) - float(expected)) <= 1e-6
        except (ValueError, TypeError):
            return False

    def get_ground_truth(self, problem: dict):
        return self.extract_answer(problem["answer"])


class MMLUProEvaluator(DatasetEvaluator):
    """MMLU_Pro dataset: extract answer letter, compare case-insensitive."""

    def extract_answer(self, text: str):
        if not text:
            return None
        text = text.strip()
        # Pattern 1: "The answer is [X]"
        match = re.search(r'[Tt]he answer is\s*\(?([A-J])\)?', text)
        if match:
            return match.group(1).upper()
        # Pattern 2: "answer is [X]"
        match = re.search(r'answer is\s*\(?([A-J])\)?', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        # Pattern 3: "FINAL ANSWER: [X]"
        match = re.search(r'FINAL ANSWER:\s*\(?([A-J])\)?', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        # Pattern 4: "Answer: [X]"
        match = re.search(r'[Aa]nswer:\s*\(?([A-J])\)?', text)
        if match:
            return match.group(1).upper()
        # Pattern 5: Standalone letter in parentheses at end
        match = re.search(r'\(([A-J])\)\s*\.?\s*$', text)
        if match:
            return match.group(1).upper()
        # Pattern 6: Last single letter on its own line
        for line in reversed(text.strip().split('\n')):
            line = line.strip().rstrip('.')
            if len(line) == 1 and line.upper() in 'ABCDEFGHIJ':
                return line.upper()
        # Pattern 7: "is [X]" fallback
        match = re.search(r'\bis\s+([A-J])\b', text)
        if match:
            return match.group(1).upper()
        return None

    def check_correct(self, predicted, expected) -> bool:
        if predicted is None or expected is None:
            return False
        return str(predicted).upper() == str(expected).upper()

    def get_ground_truth(self, problem: dict):
        return problem["answer"]

    @staticmethod
    def format_question(problem: dict) -> str:
        question = problem.get("question", "")
        options = problem.get("options", [])
        category = problem.get("category", "")
        formatted = f"The following is a multiple choice question about {category}.\n\n{question}\n\n"
        for i, option in enumerate(options):
            letter = string.ascii_uppercase[i]
            formatted += f"{letter}: {option}\n"
        return formatted.strip()


def create_evaluator(dataset: str) -> DatasetEvaluator:
    evaluators = {
        "MATH": MATHEvaluator,
        "GSM8K": GSM8KEvaluator,
        "MMLU_Pro": MMLUProEvaluator,
    }
    if dataset not in evaluators:
        raise ValueError(f"Unknown dataset: {dataset}. Supported: {list(evaluators.keys())}")
    return evaluators[dataset]()


# ============================================================
# Data loading
# ============================================================

def load_ground_truth(data_path: str, dataset: str, evaluator: DatasetEvaluator) -> dict:
    """Load dataset and build input_prefix → ground_truth lookup."""
    problems = []
    with open(data_path) as f:
        for line in f:
            try:
                problems.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue

    lookup = {}
    for p in problems:
        if dataset == "MATH":
            key = p["problem"][:200]
        elif dataset == "GSM8K":
            key = p["question"][:200]
        elif dataset == "MMLU_Pro":
            key = MMLUProEvaluator.format_question(p)[:200]
        else:
            continue

        gt = evaluator.get_ground_truth(p)
        lookup[key] = gt

    print(f"Loaded {len(lookup)} ground truth entries from {data_path}")
    return lookup


def match_ground_truth(input_text: str, lookup: dict):
    """Match a trace input to ground truth using prefix matching."""
    key = input_text[:200]
    if key in lookup:
        return lookup[key]
    # Fuzzy fallback: try first 100 chars
    short_key = input_text[:100]
    for k, v in lookup.items():
        if k[:100] == short_key:
            return v
    return None


def load_lambda_acc(lagrangian_curves_path: str) -> float:
    """Read final lambda_acc from lagrangian curves CSV."""
    last_lambda_acc = 1.0
    with open(lagrangian_curves_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            last_lambda_acc = float(row["lambda_acc"])
    print(f"Loaded lambda_acc = {last_lambda_acc:.4f} from {lagrangian_curves_path}")
    return last_lambda_acc


# ============================================================
# Trace processing
# ============================================================

TERMINAL_TYPES = {"generate", "multi_generate", "refine", "programmer"}


def get_result_type(op: dict) -> str:
    """Get result type, handling both scheduler and non-scheduler traces."""
    rt = op.get("result_type")
    if rt:
        return rt
    # Fallback: map operator name to type
    name = op.get("operator", "").lower()
    if "multi" in name and "generate" in name:
        return "multi_generate"
    if "generate" in name:
        return "generate"
    if "refine" in name:
        return "refine"
    if "programmer" in name:
        return "programmer"
    if "ensemble" in name:
        return "ensemble"
    if "test" in name:
        return "test"
    return "unknown"


def _extract_tokens_from_output(output_str: str) -> Tuple[float, float]:
    """Extract prompt and completion token counts from trace output JSON.

    Returns (prompt_tokens, completion_tokens). Falls back to (0, 0) if unparseable.
    """
    if not output_str:
        return 0.0, 0.0
    try:
        # Try to parse as JSON (works for non-truncated outputs)
        data = json.loads(output_str) if isinstance(output_str, str) else output_str
        if isinstance(data, dict):
            prompt = data.get("_usage_prompt_tokens", 0) or 0
            completion = data.get("_usage_tokens", 0) or 0
            return float(prompt), float(completion)
    except (json.JSONDecodeError, TypeError):
        pass
    return 0.0, 0.0


def extract_solution_from_output(output_str: str, result_type: str) -> Optional[str]:
    """Parse solution text from trace output field.

    Returns a single solution string, or None if unparseable.
    For multi_generate, returns a list of solutions.
    """
    if not output_str:
        return None

    is_truncated = "...[truncated]" in output_str or "...[middle truncated]..." in output_str

    # Try JSON parsing first (works for non-truncated outputs)
    try:
        data = json.loads(output_str) if isinstance(output_str, str) else output_str
    except json.JSONDecodeError:
        data = None

    if isinstance(data, dict):
        if result_type == "multi_generate":
            responses = data.get("response", data.get("solutions", []))
            if isinstance(responses, list):
                sols = []
                for r in responses:
                    if isinstance(r, dict):
                        sols.append(r.get("response", str(r)))
                    else:
                        sols.append(str(r))
                return sols if sols else None
            return None
        for key in ("response", "solution", "refined", "selected", "output"):
            if key in data:
                return data[key]
        return str(data)

    # JSON parsing failed (typically truncated output).
    # Try to extract the raw text content from the broken JSON string.
    # The format is {"response": "...text...[truncated]"} — extract text after first `"response": "`.
    if is_truncated or data is None:
        raw = output_str
        # Try to pull the value of "response" or "solution" from the broken JSON
        for key in ('"response"', '"solution"', '"refined"'):
            idx = raw.find(f'{key}: "')
            if idx < 0:
                idx = raw.find(f'{key}:"')
            if idx < 0:
                idx = raw.find(f'{key}: \\"')
            if idx >= 0:
                # Find the start of the value
                val_start = raw.index('"', idx + len(key) + 1) + 1 if '"' in raw[idx + len(key):] else -1
                if val_start > 0:
                    # Return everything from val_start to end (minus truncation marker)
                    text = raw[val_start:].rstrip('...[truncated]').rstrip('"').rstrip()
                    # Unescape JSON string escapes
                    text = text.replace('\\"', '"').replace('\\n', '\n').replace('\\\\', '\\')
                    if len(text) > 10:  # Sanity check: must have some content
                        return text
        # Last resort: return the raw string (might contain the answer somewhere)
        cleaned = raw.rstrip('...[truncated]')
        if len(cleaned) > 50:
            return cleaned

    return None


# ============================================================
# Trace loading and grouping
# ============================================================

def load_traces(trace_dir: str, dataset: str):
    """Load all operator traces and group by problem execution."""
    trace_files = sorted(Path(trace_dir).glob(f"operator_trace_{dataset}_*.jsonl"))
    if not trace_files:
        print(f"No trace files found in {trace_dir} for {dataset}")
        return []

    all_entries = []
    for tf in trace_files:
        with open(tf) as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    all_entries.append(entry)
                except json.JSONDecodeError:
                    continue

    print(f"Loaded {len(all_entries)} trace entries from {len(trace_files)} files")

    # Filter to scheduler traces only (have start_offset — from Phase 1.5 collect_traces)
    scheduler_entries = [e for e in all_entries if e.get("start_offset") is not None or e.get("layer", 0) < 0]
    if scheduler_entries:
        print(f"Filtered to {len(scheduler_entries)} scheduler entries (with start_offset) out of {len(all_entries)} total")
        all_entries = scheduler_entries
    else:
        print(f"Warning: No entries with start_offset found, using all {len(all_entries)} entries (timestamp-based T_k)")

    all_entries.sort(key=lambda x: x.get("ts", ""))

    executions = []
    current_exec = []
    last_ts = None

    for entry in all_entries:
        ts_str = entry.get("ts", "")
        problem_id = entry.get("problem_id", entry.get("input", "")[:200])

        try:
            ts = datetime.fromisoformat(ts_str)
        except (ValueError, TypeError):
            continue

        if current_exec:
            prev_problem = current_exec[-1].get("problem_id", current_exec[-1].get("input", "")[:200])
            if problem_id != prev_problem or (last_ts and (ts - last_ts).total_seconds() > 60):
                if len(current_exec) >= 1:
                    executions.append(current_exec)
                current_exec = []

        current_exec.append(entry)
        last_ts = ts

    if current_exec:
        executions.append(current_exec)

    print(f"Grouped into {len(executions)} problem executions")
    return executions


# ============================================================
# Training data generation with ground truth
# ============================================================

def generate_training_data(
    executions: list,
    evaluator: DatasetEvaluator,
    gt_lookup: dict,
    lambda_acc: float,
    acc_baseline: float,
    latency_weight: float,
    cost_weight: float,
    dataset: str = None,
):
    """Generate (features, delta) pairs using ground truth evaluation.

    R_k = lambda_acc * (S_k / acc_baseline) - latency_weight * T_k - cost_weight * C_k
    Delta_k = R_k - R_N

    All parameters (lambda_acc, acc_baseline, latency_weight, cost_weight) are
    reused from the orchestrator's Lagrangian training. No extra hyperparameters.
    """
    all_features = []
    all_deltas = []
    stats = {
        "total_execs": 0, "usable_execs": 0, "no_gt": 0,
        "total_samples": 0, "positive_delta": 0, "negative_delta": 0,
    }

    # Cache query embeddings by problem text prefix (avoid re-encoding same problem)
    _emb_cache = {}

    for exec_ops in executions:
        stats["total_execs"] += 1

        # Filter to DAG operators (layer >= 0)
        dag_ops = [op for op in exec_ops if op.get("layer", -1) >= 0]
        if len(dag_ops) < 2:
            continue

        # Match to ground truth
        input_text = exec_ops[0].get("input", "")
        gt = match_ground_truth(input_text, gt_lookup)
        if gt is None:
            stats["no_gt"] += 1
            continue

        # Compute query embedding (cached by problem text prefix)
        emb_key = input_text[:200]
        if emb_key not in _emb_cache:
            _emb_cache[emb_key] = get_sentence_embedding(input_text[:1000]).tolist()
        query_emb = _emb_cache[emb_key]

        # Sort by completion time
        for op in dag_ops:
            start_offset = op.get("start_offset", None)
            if start_offset is not None:
                op["_finish"] = start_offset + op.get("latency", 0)
            else:
                try:
                    ts = datetime.fromisoformat(op["ts"])
                    op["_finish"] = ts.timestamp()
                except Exception:
                    op["_finish"] = 0
        dag_ops.sort(key=lambda x: x["_finish"])

        base_finish = dag_ops[0]["_finish"] - dag_ops[0].get("latency", 0)
        for op in dag_ops:
            op["_finish_rel"] = op["_finish"] - base_finish

        n_total = len(dag_ops)

        # --- Phase A: extract all solutions and pre-evaluate correctness ---
        per_op_sol = []
        unique_sols = {}
        for op in dag_ops:
            result_type = get_result_type(op)
            sol = extract_solution_from_output(op.get("output", ""), result_type)
            per_op_sol.append((result_type, sol))
            if sol is not None:
                if isinstance(sol, list):
                    for s in sol:
                        h = _consensus_key(str(s), dataset)
                        if h not in unique_sols:
                            unique_sols[h] = s
                else:
                    h = _consensus_key(str(sol), dataset)
                    if h not in unique_sols:
                        unique_sols[h] = sol

        sol_correct = {}
        for h, s in unique_sols.items():
            sol_correct[h] = evaluator.check_correct(evaluator.extract_answer(s), gt)

        # --- Phase B: simulate cumulative state at each stop point ---
        terminal_solutions_hashes = []
        solution_hash_counts = {}
        n_completed = 0
        n_solutions = 0
        cumulative_cost = 0.0

        stop_points = []

        for i, op in enumerate(dag_ops):
            n_completed += 1
            result_type, sol = per_op_sol[i]

            if sol is not None:
                if result_type == "ensemble":
                    s = sol if isinstance(sol, str) else sol[0]
                    h = _consensus_key(str(s), dataset)
                    terminal_solutions_hashes = [h]
                    solution_hash_counts = {h: 1}
                    n_solutions = 1
                elif result_type == "multi_generate" and isinstance(sol, list):
                    for s in sol:
                        h = _consensus_key(str(s), dataset)
                        terminal_solutions_hashes.append(h)
                        solution_hash_counts[h] = solution_hash_counts.get(h, 0) + 1
                    n_solutions += len(sol)
                else:
                    h = _consensus_key(str(sol), dataset)
                    terminal_solutions_hashes.append(h)
                    solution_hash_counts[h] = solution_hash_counts.get(h, 0) + 1
                    n_solutions += 1

            n_unique = len(solution_hash_counts)
            max_agree = max(solution_hash_counts.values()) if solution_hash_counts else 0
            n_pending = n_total - n_completed

            T_k = op["_finish_rel"]
            # Cost: extract prompt + completion tokens from trace output, use gpt-4o-mini pricing
            # Matches controller: (prompt * 0.00015 + completion * 0.0006) / 1000
            prompt_tok, completion_tok = _extract_tokens_from_output(op.get("output", ""))
            cumulative_cost += (prompt_tok * 0.00015 + completion_tok * 0.0006) / 1000

            # Correctness at this stop point: majority vote on extracted answers,
            # then check if the majority is correct
            if n_solutions > 0:
                answer_counts = {}
                for h in terminal_solutions_hashes:
                    s = unique_sols.get(h, "")
                    ans = evaluator.extract_answer(s)
                    if ans is None:
                        continue
                    ans_key = str(ans)
                    if ans_key not in answer_counts:
                        answer_counts[ans_key] = [0, sol_correct.get(h, False)]
                    answer_counts[ans_key][0] += 1
                if answer_counts:
                    majority_ans = max(answer_counts.items(), key=lambda x: x[1][0])
                    correctness_k = 1.0 if majority_ans[1][1] else 0.0
                else:
                    correctness_k = 0.0
            else:
                correctness_k = 0.0

            stop_points.append({
                "n_completed": n_completed,
                "n_solutions": n_solutions,
                "max_agree": max_agree,
                "n_unique": n_unique,
                "n_pending": n_pending,
                "T_k": T_k,
                "C_k": cumulative_cost,
                "S_k": correctness_k,
            })

        if not stop_points or stop_points[-1]["n_solutions"] < 1:
            continue

        # Final step values (baseline: all operators executed)
        T_N = stop_points[-1]["T_k"]
        C_N = stop_points[-1]["C_k"]
        S_N = stop_points[-1]["S_k"]
        norm_S_N = S_N / acc_baseline if acc_baseline > 0 else S_N
        R_N = lambda_acc * norm_S_N - latency_weight * T_N - cost_weight * C_N

        # Generate training samples
        for sp in stop_points:
            if sp["n_solutions"] < 1:
                continue
            if sp["n_pending"] == 0:
                continue

            # R_k = lambda_acc * (S_k / S_0) - lambda_t * T_k - lambda_c * C_k
            norm_S_k = sp["S_k"] / acc_baseline if acc_baseline > 0 else sp["S_k"]
            R_k = lambda_acc * norm_S_k - latency_weight * sp["T_k"] - cost_weight * sp["C_k"]
            delta_k = R_k - R_N

            # State features (2-dim) + query embedding (384-dim, stored separately)
            state_features = [
                sp["n_completed"] / max(n_total, 1),                                  # progress
                sp["max_agree"] / max(sp["n_solutions"], 1) if sp["n_solutions"] > 0 else 0,  # consensus
            ]
            features = (state_features, query_emb)  # tuple: (6-dim list, 384-dim list)

            all_features.append(features)
            all_deltas.append(delta_k)
            stats["total_samples"] += 1
            if delta_k >= 0:
                stats["positive_delta"] += 1
            else:
                stats["negative_delta"] += 1

        stats["usable_execs"] += 1

    print(f"\nStats: {stats}")
    if stats["total_samples"] > 0:
        pos = stats["positive_delta"]
        neg = stats["negative_delta"]
        print(f"Delta distribution: positive={pos} ({pos/(pos+neg):.1%}), negative={neg} ({neg/(pos+neg):.1%})")
        deltas_arr = all_deltas
        print(f"Delta range: [{min(deltas_arr):.3f}, {max(deltas_arr):.3f}], mean={sum(deltas_arr)/len(deltas_arr):.3f}")

    return all_features, all_deltas


# ============================================================
# Training
# ============================================================

def train_gate(features, deltas, hidden_dim=32, epochs=100, lr=0.01):
    """Train PruningGate with MSE loss on delta_k. Returns (gate, tau).

    features: list of (state_list, query_emb_list) tuples
    tau = std({delta_k | delta_k > 0}) — temperature for inference sigmoid.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_state = torch.tensor([f[0] for f in features], dtype=torch.float32, device=device)
    X_query = torch.tensor([f[1] for f in features], dtype=torch.float32, device=device)
    y = torch.tensor(deltas, dtype=torch.float32, device=device)

    gate = PruningGate(hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(gate.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        gate.train()
        pred = gate(X_state, X_query)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            mae = (pred - y).abs().mean().item()
            # Classification accuracy: does sign(pred) match sign(delta)?
            sign_acc = ((pred > 0) == (y > 0)).float().mean().item()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.cpu().clone() for k, v in gate.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:4d} | MSE: {loss.item():.4f} | MAE: {mae:.4f} | SignAcc: {sign_acc:.4f}")

    gate.load_state_dict(best_state)
    gate.eval()

    # Compute tau = std of positive deltas
    positive_deltas = [d for d in deltas if d > 0]
    if len(positive_deltas) >= 2:
        mean_pos = sum(positive_deltas) / len(positive_deltas)
        var_pos = sum((d - mean_pos) ** 2 for d in positive_deltas) / (len(positive_deltas) - 1)
        tau = var_pos ** 0.5
    else:
        tau = 0.1  # fallback
    tau = max(tau, 0.01)  # prevent division by near-zero

    print(f"\nBest MSE: {best_loss:.4f}")
    print(f"Tau (std of positive deltas): {tau:.4f} (from {len(positive_deltas)} positive samples)")

    return gate, tau


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train pruning gate from execution traces")
    # I/O
    parser.add_argument("--trace_dir", type=str, required=True,
                        help="Directory containing operator_trace_*.jsonl files")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (MATH, GSM8K, MMLU_Pro)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to dataset JSONL file with ground truth")
    parser.add_argument("--output_path", type=str, default="pruning_gate.pth",
                        help="Output path for trained gate checkpoint")

    # Reward parameters (reused from orchestrator's Lagrangian training)
    parser.add_argument("--latency_weight", type=float, default=0.005,
                        help="Latency penalty weight lambda_t (same as controller training)")
    parser.add_argument("--cost_weight", type=float, default=3.0,
                        help="Cost penalty weight lambda_c (same as controller training)")
    parser.add_argument("--acc_baseline", type=float, default=0.8,
                        help="Accuracy baseline S_0 for reward normalization")
    parser.add_argument("--lambda_acc", type=float, default=None,
                        help="Lambda_acc value (overrides --lagrangian_curves)")
    parser.add_argument("--lagrangian_curves", type=str, default=None,
                        help="Path to lagrangian curves CSV to read final lambda_acc")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--hidden_dim", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve lambda_acc
    if args.lambda_acc is not None:
        lambda_acc = args.lambda_acc
    elif args.lagrangian_curves:
        lambda_acc = load_lambda_acc(args.lagrangian_curves)
    else:
        lambda_acc = 1.0
        print("Warning: No lambda_acc specified, using default 1.0")

    print(f"=== Training Pruning Gate (Delta Regression) ===")
    print(f"Trace dir:      {args.trace_dir}")
    print(f"Dataset:        {args.dataset}")
    print(f"Data path:      {args.data_path}")
    print(f"Lambda_acc:     {lambda_acc:.4f}")
    print(f"Acc baseline:   {args.acc_baseline}")
    print(f"Latency weight: {args.latency_weight}")
    print(f"Cost weight:    {args.cost_weight}")
    print()

    # Step 1: Create evaluator
    evaluator = create_evaluator(args.dataset)

    # Step 2: Load ground truth
    gt_lookup = load_ground_truth(args.data_path, args.dataset, evaluator)

    # Step 3: Load traces
    executions = load_traces(args.trace_dir, args.dataset)
    if not executions:
        print("No execution data found. Exiting.")
        return

    # Step 4: Generate training data with ground truth
    features, deltas = generate_training_data(
        executions, evaluator, gt_lookup,
        lambda_acc=lambda_acc,
        acc_baseline=args.acc_baseline,
        latency_weight=args.latency_weight,
        cost_weight=args.cost_weight,
        dataset=args.dataset,
    )
    if not features:
        print("No training samples generated. Exiting.")
        return

    print(f"\nTraining samples: {len(features)}")
    print(f"Feature dim: {len(features[0])}")
    print()

    # Step 5: Train gate
    gate, tau = train_gate(features, deltas, hidden_dim=args.hidden_dim, epochs=args.epochs, lr=args.lr)

    # Step 6: Save (gate weights + tau)
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    torch.save({"gate_state_dict": gate.state_dict(), "tau": tau}, args.output_path)
    print(f"\nGate saved to {args.output_path}")
    print(f"Parameters: {sum(p.numel() for p in gate.parameters())}")
    print(f"Tau: {tau:.4f}")


if __name__ == "__main__":
    main()
