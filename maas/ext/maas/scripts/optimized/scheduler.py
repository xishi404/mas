"""
Operator Graph Scheduler for MaAS.

Reorders the controller's layer-by-layer operator assignments so that all
generators (stateless operators that only depend on the problem) are hoisted
to Scheduled Layer 0 and run in parallel, while dependent operators
(SelfRefine, Test, Debate, ScEnsemble) are placed in subsequent layers with
explicit dependency tracking.

The scheduler preserves semantic equivalence with the original serial
(left-to-right, layer-by-layer) execution order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Operator classification
# ---------------------------------------------------------------------------
# Generators: only depend on `problem`; safe to hoist to Scheduled Layer 0.
GENERATORS = {"Generate", "GenerateCoT", "MultiGenerateCoT", "Programmer"}

# Dependents: need `current_solution` or `solutions` from prior operators.
DEPENDENTS = {"SelfRefine", "Test", "Debate", "ScEnsemble"}

# Noop operators that are simply skipped.
SKIP = {"EarlyStop"}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class ScheduledTask:
    """A single operator invocation in the scheduled plan."""

    task_id: str                # unique id, e.g. "Generate#0", "SelfRefine#1"
    operator_name: str          # registry name, e.g. "Generate"
    scheduled_layer: int        # 0 for generators, ≥1 for dependents
    original_layer: int         # index in the controller's selected_names_layers
    original_position: int      # position within that original layer

    # --- dependency metadata ---
    # For dependents: task_id whose output to use as `current_solution`.
    depends_on_solution: Optional[str] = None

    # For ScEnsemble only: ordered list of task_ids whose outputs form the
    # `solutions` list.  Each entry is a task_id; for MultiGenerateCoT the
    # executor knows to expand its 3 sub-outputs.
    depends_on_solutions: Optional[List[str]] = None


@dataclass
class ScheduledPlan:
    """The full scheduled execution plan."""

    layers: List[List[ScheduledTask]]   # layers[0] = generators, layers[1+] = dependents
    original_layers: List[List[str]]    # preserved original selected_names_layers
    task_index: dict = field(default_factory=dict)  # task_id -> ScheduledTask


# ---------------------------------------------------------------------------
# Scheduling algorithm
# ---------------------------------------------------------------------------
def schedule(selected_names_layers: List[List[str]]) -> ScheduledPlan:
    """
    Transform the controller's operator selection into a scheduled plan.

    Assumes the original execution is serial: layer-by-layer, left-to-right
    within each layer.  The scheduler:

    1. Flattens all operators into serial order.
    2. Simulates state tracking (current_solution_source, solutions_sources)
       to record each operator's dependencies.
    3. Assigns generators to Scheduled Layer 0.
    4. Assigns dependents to layers determined by topological ordering of
       their dependencies.
    5. Groups tasks by scheduled layer.
    """

    # --- Step 1: flatten and assign task IDs ---
    task_counter: dict[str, int] = {}
    all_tasks: List[ScheduledTask] = []

    # State simulation
    current_solution_source: Optional[str] = None
    solutions_sources: List[str] = []  # task_ids contributing to solutions[]

    for layer_idx, names in enumerate(selected_names_layers):
        for pos, op_name in enumerate(names):
            if op_name in SKIP:
                continue

            count = task_counter.get(op_name, 0)
            task_id = f"{op_name}#{count}"
            task_counter[op_name] = count + 1

            if op_name in GENERATORS:
                task = ScheduledTask(
                    task_id=task_id,
                    operator_name=op_name,
                    scheduled_layer=0,  # hoisted
                    original_layer=layer_idx,
                    original_position=pos,
                    depends_on_solution=None,
                    depends_on_solutions=None,
                )
                # Simulate state update
                solutions_sources.append(task_id)
                current_solution_source = task_id

            elif op_name == "ScEnsemble":
                task = ScheduledTask(
                    task_id=task_id,
                    operator_name=op_name,
                    scheduled_layer=1,  # tentative; refined in step 3
                    original_layer=layer_idx,
                    original_position=pos,
                    depends_on_solution=current_solution_source,
                    depends_on_solutions=list(solutions_sources),  # snapshot
                )
                # ScEnsemble resets solutions
                solutions_sources = [task_id]
                current_solution_source = task_id

            else:
                # SelfRefine, Test, Debate
                task = ScheduledTask(
                    task_id=task_id,
                    operator_name=op_name,
                    scheduled_layer=1,  # tentative
                    original_layer=layer_idx,
                    original_position=pos,
                    depends_on_solution=current_solution_source,
                    depends_on_solutions=None,
                )
                solutions_sources.append(task_id)
                current_solution_source = task_id

            all_tasks.append(task)

    # --- Step 2: build task index ---
    task_index: dict[str, ScheduledTask] = {t.task_id: t for t in all_tasks}

    # --- Step 3: refine scheduled layers via topological ordering ---
    # A dependent's layer must be > all of its dependencies' layers.
    # Generators are fixed at layer 0.  Iterate until stable.
    changed = True
    while changed:
        changed = False
        for task in all_tasks:
            if task.scheduled_layer == 0:
                continue  # generators stay at 0

            min_layer = 1  # at least layer 1

            # depends_on_solution
            if task.depends_on_solution and task.depends_on_solution in task_index:
                dep = task_index[task.depends_on_solution]
                min_layer = max(min_layer, dep.scheduled_layer + 1)

            # depends_on_solutions (ScEnsemble)
            if task.depends_on_solutions:
                for dep_id in task.depends_on_solutions:
                    if dep_id in task_index:
                        dep = task_index[dep_id]
                        min_layer = max(min_layer, dep.scheduled_layer + 1)

            if min_layer > task.scheduled_layer:
                task.scheduled_layer = min_layer
                changed = True

    # --- Step 4: group into layers ---
    max_layer = max((t.scheduled_layer for t in all_tasks), default=0)
    layers: List[List[ScheduledTask]] = [[] for _ in range(max_layer + 1)]
    for task in all_tasks:
        layers[task.scheduled_layer].append(task)

    return ScheduledPlan(
        layers=layers,
        original_layers=selected_names_layers,
        task_index=task_index,
    )
