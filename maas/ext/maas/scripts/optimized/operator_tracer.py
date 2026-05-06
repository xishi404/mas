"""Operator-level trace logger. Appends one JSONL record per operator execution."""

import json
import os
import threading
from datetime import datetime


class OperatorTracer:
    def __init__(self, log_dir: str, dataset: str):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(log_dir, f"operator_trace_{dataset}_{timestamp}.jsonl")
        self.dataset = dataset
        self._lock = threading.Lock()

    def log(self, *, layer: int, operator: str, input: str, output, latency: float = 0.0, tokens: int = 0,
            task_id: str = None, start_offset: float = None, problem_id: str = None, result_type: str = None, extra: dict = None):
        record = {
            "ts": datetime.now().isoformat(),
            "dataset": self.dataset,
            "layer": layer,
            "operator": operator,
            "input": _truncate(input, 2000),
            "output": _truncate_keep_tail(_to_str(output), 4000),
            "latency": round(latency, 3),
            "tokens": tokens,
        }
        if task_id is not None:
            record["task_id"] = task_id
        if start_offset is not None:
            record["start_offset"] = round(start_offset, 3)
        if problem_id is not None:
            record["problem_id"] = problem_id
        if result_type is not None:
            record["result_type"] = result_type
        if extra:
            record.update(extra)
        with self._lock:
            with open(self.path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _truncate(s: str, max_len: int) -> str:
    return s if len(s) <= max_len else s[:max_len] + "...[truncated]"


def _truncate_keep_tail(s: str, max_len: int) -> str:
    """Truncate keeping head and tail, omitting middle. Preserves answer at end."""
    if len(s) <= max_len:
        return s
    # Keep head and tail, each gets ~half the budget minus marker
    marker = "...[middle truncated]..."
    budget = max_len - len(marker)
    head = budget // 3       # 1/3 for head (less important)
    tail = budget - head     # 2/3 for tail (contains answer)
    return s[:head] + marker + s[-tail:]


def _to_str(obj) -> str:
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return json.dumps(obj, ensure_ascii=False, default=str)
    if isinstance(obj, list):
        return json.dumps(obj, ensure_ascii=False, default=str)
    return str(obj)
