"""Shared DAG executor for scheduled operator execution.

Used by all dataset graphs (GSM8K, MATH, MMLU_Pro). Handles:
  - DAG dependency tracking via asyncio.Event
  - Parallel task execution via asyncio.gather
  - Operator-level trace logging
  - Pruning gate integration
  - Critical path computation
  - Metrics collection

Each dataset provides its own `dispatch_fn` that handles operator-specific
calling conventions (prompts, arguments, etc.).
"""

import asyncio
import hashlib
import random
import re
import time
from typing import Callable, Dict, List, Tuple

import torch

from lamas.logs import logger


TERMINAL_TYPES = {"generate", "multi_generate", "refine", "programmer"}


# --- Dataset-specific answer extractors for semantic consensus ---
# Two solutions are considered "agreeing" when their extracted answers match.
# Falls back to MD5 of full text when no extractor is registered.

def _extract_math_answer(text):
    """Extract \\boxed{...} content from a MATH solution."""
    if not text:
        return None
    matches = re.findall(r"\\boxed{((?:[^{}]|{[^{}]*})*)}", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None


def _extract_gsm8k_answer(text):
    """Extract last number from a GSM8K solution."""
    if not text:
        return None
    matches = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?|\d+\.\d+", str(text))
    if matches:
        try:
            return str(float(matches[-1].replace(",", "")))
        except ValueError:
            return None
    return None


def _extract_mmlu_answer(text):
    """Extract answer letter (A-J) from an MMLU_Pro solution."""
    if not text:
        return None
    text = text.strip()
    patterns = [
        r'[Tt]he answer is\s*\(?([A-J])\)?',
        r'answer is\s*\(?([A-J])\)?',
        r'FINAL ANSWER:\s*\(?([A-J])\)?',
        r'[Aa]nswer:\s*\(?([A-J])\)?',
        r'\(([A-J])\)\s*\.?\s*$',
        r'\bis\s+([A-J])\b',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE if "answer is" in pat.lower() else 0)
        if m:
            return m.group(1).upper()
    return None


DATASET_EXTRACTORS = {
    "MATH": _extract_math_answer,
    "GSM8K": _extract_gsm8k_answer,
    "MMLU_Pro": _extract_mmlu_answer,
}


def _consensus_key(solution_text, dataset):
    """Compute the consensus grouping key for a solution.

    Uses the dataset-specific answer extractor when available; falls back to
    MD5 hash of the full solution text otherwise.
    """
    extractor = DATASET_EXTRACTORS.get(dataset)
    if extractor:
        ans = extractor(solution_text)
        if ans is not None:
            return f"ans:{ans}"
    return hashlib.md5(solution_text.strip().encode()).hexdigest()


async def execute_dag(
    plan,
    problem: str,
    dispatch_fn: Callable,
    tracer,
    device: torch.device,
    pruning_gate=None,
    problem_id: str = None,
    dataset: str = None,
) -> Tuple[Dict, float, List, bool, Dict]:
    """Execute a scheduled DAG plan with fine-grained dependency tracking.

    Args:
        plan: ScheduledPlan from scheduler.schedule()
        problem: The problem text
        dispatch_fn: async fn(task, dep_solution, task_outputs) -> (raw_result, ret_dict)
        tracer: OperatorTracer instance or None
        device: torch device
        pruning_gate: Learned PruningGate nn.Module, or None to disable
        problem_id: For trace logging
        dataset: Dataset name (used for semantic consensus extraction)

    Returns:
        (task_outputs, total_time, gate_log_probs, pruning_triggered, gate_state)
    """
    task_outputs = {}
    completion_events = {tid: asyncio.Event() for tid in plan.task_index}
    exec_start = time.time()

    pruning_event = asyncio.Event()
    pruning_triggered = [False]
    gate_log_probs = []
    _gate_state = {
        "n_completed": 0, "n_solutions": 0, "n_unique": 0,
        "solution_hashes": {}, "n_total": len(plan.task_index),
    }

    # Pre-compute query embedding for the gate (computed once per problem)
    _query_emb = None
    if pruning_gate is not None:
        from lamas.ext.lamas.models.utils import get_sentence_embedding
        _query_emb = get_sentence_embedding(problem[:1000]).to(device)

    async def execute_task(task):
        op_name = task.operator_name

        # Wait for direct dependencies
        dep_ids = set()
        if task.depends_on_solution:
            dep_ids.add(task.depends_on_solution)
        if task.depends_on_solutions:
            dep_ids.update(task.depends_on_solutions)
        for dep_id in dep_ids:
            if dep_id in completion_events:
                await completion_events[dep_id].wait()

        # Skip if pruning has been triggered and this task is still pending
        if pruning_event.is_set():
            task_start = time.time()
            ret = {
                "type": "noop", "solution": "", "operator": op_name,
                "latency": 0.0, "iterations": 0, "cp_token_count": 0,
                "_raw_prompt_tokens": 0, "_raw_completion_tokens": 0,
                "_task_id": task.task_id, "_start_offset": task_start - exec_start,
                "_skipped_pruning": True,
            }
            task_outputs[task.task_id] = ret
            completion_events[task.task_id].set()
            return ret

        task_start = time.time()
        start_offset = task_start - exec_start

        # Resolve dependency solution
        dep_solution = ""
        if task.depends_on_solution and task.depends_on_solution in task_outputs:
            dep_out = task_outputs[task.depends_on_solution]
            dep_solution = dep_out.get("solution", "")
            if dep_out.get("type") == "multi_generate" and dep_out.get("solutions"):
                dep_solution = dep_out["solutions"][-1]

        raw_result, ret = await dispatch_fn(task, dep_solution, task_outputs)

        latency = time.time() - task_start
        if "latency" not in ret or ret["latency"] == 0:
            ret["latency"] = latency
        ret["_task_id"] = task.task_id
        ret["_start_offset"] = start_offset

        if tracer:
            tracer.log(
                layer=task.original_layer, operator=op_name,
                input=problem, output=raw_result,
                latency=ret.get("latency", 0), tokens=ret.get("cp_token_count", 0),
                task_id=task.task_id, start_offset=start_offset,
                problem_id=problem_id or problem[:100],
                result_type=ret.get("type", ""),
            )

        # Gate-based pruning
        if pruning_gate is not None and not pruning_event.is_set():
            _gate_state["n_completed"] += 1

            sol_text = ret.get("solution", "")
            if ret.get("type") == "ensemble" and sol_text:
                h = _consensus_key(sol_text, dataset)
                _gate_state["solution_hashes"] = {h: 1}
                _gate_state["n_solutions"] = 1
                _gate_state["n_unique"] = 1
            elif ret.get("type") == "multi_generate":
                for s in ret.get("solutions", []):
                    h = _consensus_key(s, dataset)
                    _gate_state["solution_hashes"][h] = _gate_state["solution_hashes"].get(h, 0) + 1
                    _gate_state["n_solutions"] += 1
                _gate_state["n_unique"] = len(_gate_state["solution_hashes"])
            elif sol_text:
                h = _consensus_key(sol_text, dataset)
                _gate_state["solution_hashes"][h] = _gate_state["solution_hashes"].get(h, 0) + 1
                _gate_state["n_solutions"] += 1
                _gate_state["n_unique"] = len(_gate_state["solution_hashes"])

            n_t = _gate_state["n_total"]
            n_comp = _gate_state["n_completed"]
            n_sols = _gate_state["n_solutions"]
            max_agree = max(_gate_state["solution_hashes"].values()) if _gate_state["solution_hashes"] else 0
            n_pending = sum(1 for ev in completion_events.values() if not ev.is_set())

            if n_sols >= 1 and _query_emb is not None:
                consensus = max_agree / max(n_sols, 1)
                state_features = torch.tensor([
                    n_comp / max(n_t, 1),
                    consensus,
                ], device=device, dtype=torch.float32)
                with torch.no_grad():
                    delta_hat = pruning_gate(state_features, _query_emb)
                tau = getattr(pruning_gate, '_tau', 0.1)
                p_halt = torch.sigmoid(delta_hat / tau)
                halt = random.random() < p_halt.item()
                gate_log_probs.append(
                    torch.log(p_halt + 1e-10) if halt else torch.log(1 - p_halt + 1e-10)
                )

                if halt and n_pending > 0:
                    pruning_event.set()
                    pruning_triggered[0] = True

        task_outputs[task.task_id] = ret
        completion_events[task.task_id].set()
        return ret

    await asyncio.gather(*[execute_task(t) for t in plan.task_index.values()])
    total_time = time.time() - exec_start

    if pruning_triggered[0]:
        skipped = sum(1 for r in task_outputs.values() if r.get("_skipped_pruning"))
        n_terms = _gate_state["n_solutions"]
        max_agree = max(_gate_state["solution_hashes"].values()) if _gate_state["solution_hashes"] else 0
        consensus = max_agree / max(n_terms, 1)
        logger.info(f"  Gate pruning: terminals={n_terms}, consensus={consensus:.2f}, skipped={skipped}/{len(plan.task_index)} tasks")

    return task_outputs, total_time, gate_log_probs, pruning_triggered[0], _gate_state


def compute_critical_path(plan, task_outputs):
    """Compute the longest path through the DAG by finish time."""
    finish_times = {}
    for task in sorted(plan.task_index.values(), key=lambda t: (t.original_layer, t.original_position)):
        out = task_outputs.get(task.task_id)
        if not out:
            continue
        finish_times[task.task_id] = out.get("_start_offset", 0) + out.get("latency", 0)

    if not finish_times:
        return 0.0, [], 0.0

    cp_finish = max(finish_times.values())
    cp_task_id = max(finish_times, key=finish_times.get)
    chain = [cp_task_id]
    return cp_finish, chain, cp_finish


def compute_path_lengths(plan, task_outputs):
    """For each task, return the length of the longest start-to-end path passing through it.

    longest_path_through(i) = earliest_finish(i) + longest_chain_after(i)
      earliest_finish(i)    = start_offset(i) + latency(i)            (already measured)
      longest_chain_after(i) = max over successors k of (latency(k) + longest_chain_after(k))

    Tasks on the actual critical path get path_length == T (= max over all tasks).
    Off-critical tasks have path_length < T proportional to their slack.
    """
    succs = {tid: [] for tid in plan.task_index}
    for tid, task in plan.task_index.items():
        for dep in [task.depends_on_solution] + (task.depends_on_solutions or []):
            if dep and dep in succs:
                succs[dep].append(tid)

    latency = {}
    finish = {}
    for tid, out in task_outputs.items():
        latency[tid] = out.get("latency", 0.0)
        finish[tid] = out.get("_start_offset", 0.0) + latency[tid]

    chain_after = {}

    def _chain_after(tid):
        if tid in chain_after:
            return chain_after[tid]
        if not succs.get(tid):
            chain_after[tid] = 0.0
            return 0.0
        chain_after[tid] = max(latency.get(k, 0.0) + _chain_after(k) for k in succs[tid])
        return chain_after[tid]

    for tid in plan.task_index:
        _chain_after(tid)

    return {tid: finish.get(tid, 0.0) + chain_after.get(tid, 0.0)
            for tid in plan.task_index}


def collect_metrics(plan, task_outputs, selection_operator_names):
    """Aggregate per-operator metrics from task outputs."""
    operator_latencies = {}
    operator_iterations = {}
    operator_names_per_layer = []
    operator_latencies_per_layer = []
    operator_path_lengths_per_layer = []
    operator_token_counts_per_layer = []
    problem_prompt_tokens = 0
    problem_completion_tokens = 0

    path_lengths = compute_path_lengths(plan, task_outputs)

    for layer in plan.layers:
        if not layer:
            continue
        layer_names, layer_lats, layer_paths, layer_toks = [], [], [], []
        for task in layer:
            result = task_outputs.get(task.task_id, {})
            op_name = result.get("operator", task.operator_name)
            layer_names.append(op_name)
            layer_lats.append(result.get("latency", 0.0))
            layer_paths.append(path_lengths.get(task.task_id, 0.0))
            layer_toks.append(result.get("cp_token_count", 0))
            problem_prompt_tokens += result.get("_raw_prompt_tokens", 0)
            problem_completion_tokens += result.get("_raw_completion_tokens", 0)
            operator_latencies.setdefault(op_name, []).append(result.get("latency", 0.0))
            operator_iterations.setdefault(op_name, []).append(result.get("iterations", 1))
        operator_names_per_layer.append(layer_names)
        operator_latencies_per_layer.append(layer_lats)
        operator_path_lengths_per_layer.append(layer_paths)
        operator_token_counts_per_layer.append(layer_toks)

    return {
        "operator_latencies": operator_latencies,
        "operator_iterations": operator_iterations,
        "operator_names_per_layer": operator_names_per_layer,
        "operator_latencies_per_layer": operator_latencies_per_layer,
        "operator_path_lengths_per_layer": operator_path_lengths_per_layer,
        "operator_token_counts_per_layer": operator_token_counts_per_layer,
        "problem_prompt_tokens": problem_prompt_tokens,
        "problem_completion_tokens": problem_completion_tokens,
    }


def reconstruct_solutions(plan, task_outputs):
    """Reconstruct the solutions list and current_solution by replaying serial order."""
    solutions = []
    current_solution = ""
    for task in sorted(plan.task_index.values(), key=lambda t: (t.original_layer, t.original_position)):
        result = task_outputs.get(task.task_id)
        if not result:
            continue
        if result["type"] == "generate":
            solutions.append(result["solution"])
            current_solution = result["solution"]
        elif result["type"] == "multi_generate":
            solutions.extend(result.get("solutions", []))
            if result.get("solutions"):
                current_solution = result["solutions"][-1]
        elif result["type"] in ("refine", "programmer", "test"):
            solutions.append(result["solution"])
            current_solution = result["solution"]
        elif result["type"] == "ensemble":
            solutions = [result["solution"]]
            current_solution = result["solution"]

    return solutions, current_solution


def rebuild_log_probs(plan, probs_layers, selection_operator_names, device):
    """Rebuild log_probs_per_layer from scheduled layers for REINFORCE."""
    sched_log_probs_per_layer = []
    for sched_layer in plan.layers:
        if not sched_layer:
            sched_log_probs_per_layer.append(torch.tensor([], device=device))
            continue
        layer_log_probs = []
        for task in sched_layer:
            orig_layer = task.original_layer
            if orig_layer < len(probs_layers):
                probs = probs_layers[orig_layer]
                op_idx = selection_operator_names.index(task.operator_name)
                layer_log_probs.append(torch.log(probs[op_idx] + 1e-10))
        if layer_log_probs:
            sched_log_probs_per_layer.append(torch.stack(layer_log_probs))
        else:
            sched_log_probs_per_layer.append(torch.tensor([], device=device))
    return sched_log_probs_per_layer
