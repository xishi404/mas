import asyncio
import torch
import maas.ext.maas.scripts.optimized.GSM8K.train.template.prompt as prompt_custom
import maas.ext.maas.scripts.optimized.GSM8K.train.template.prompt_short as prompt_short
import maas.ext.maas.scripts.optimized.GSM8K.train.template.operator as operator
import maas.ext.maas.scripts.optimized.GSM8K.train.template.op_prompt_short as op_prompt_short
from maas.ext.maas.scripts.optimized.GSM8K.train.template.operator_registry import operator_mapping, operator_names
from maas.provider.llm_provider_registry import create_llm_instance
from maas.utils.cost_manager import CostManager
from maas.logs import logger
from maas.ext.maas.scripts.optimized.operator_tracer import OperatorTracer
from maas.utils.token_counter import TOKEN_COSTS

class GSM8KGraph:
    def __init__(
        self,
        name: str,
        llm_config,
        dataset,
        controller: torch.nn.Module,
        operator_embeddings,
        parallel_execution: bool = True,
        test_operator_names=None,
        short_prompt: bool = False,
        use_scheduler: bool = True,
        use_test_selection: bool = False,
        pruning_gate=None,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.short_prompt = short_prompt
        self.use_scheduler = use_scheduler
        self.prompt_module = prompt_short if short_prompt else prompt_custom
        self.llm = create_llm_instance(llm_config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm.cost_manager = CostManager()
        self.custom = operator.Generate(self.llm)
        self.programmer = operator.Programmer(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)

        self.controller = controller.to(self.device)
        self.operator_embeddings = operator_embeddings.to(self.device)
        self.selection_operator_instances = {
            operator_name: operator_mapping[operator_name](self.llm)
            for operator_name in operator_names
        }
        self.selection_operator_names = operator_names
        self.parallel_execution = parallel_execution
        self.tracer = None  # initialized lazily in __call__ when log_path is known

        # Override operator prompts with short versions when enabled
        if short_prompt:
            logger.info("Using SHORT prompts for GSM8K operators")
            self._apply_short_prompts(op_prompt_short)

    def _apply_short_prompts(self, short_module):
        """Override operator instance prompt variables with short versions."""
        for op in list(self.selection_operator_instances.values()) + [self.programmer, self.sc_ensemble]:
            if hasattr(op, 'cot_prompt'):
                op.cot_prompt = short_module.GENERATE_COT_PROMPT
            if hasattr(op, 'ensemble_prompt'):
                op.ensemble_prompt = short_module.SC_ENSEMBLE_PROMPT
            if hasattr(op, 'selfrefine_prompt'):
                op.selfrefine_prompt = short_module.SELFREFINE_PROMPT
            if hasattr(op, 'code_verifier_prompt'):
                op.code_verifier_prompt = short_module.PYTHON_CODE_VERIFIER_PROMPT

    def _log_operator_probabilities(self, probs_layers, selected_names_layers, log_path, problem):
        """Log the probability distribution for each operator in each layer"""
        logger.info(f"\n{'='*80}")
        logger.info("OPERATOR SELECTION PROBABILITIES")
        logger.info(f"{'='*80}")

        for layer_idx, probs in enumerate(probs_layers):
            logger.info(f"\nLayer {layer_idx + 1}:")
            logger.info(f"Selected operators: {', '.join(selected_names_layers[layer_idx])}")
            logger.info(f"{'Operator':<25} {'Probability':<15} {'Selected':<10}")
            logger.info("-" * 50)

            probs_cpu = probs.detach().cpu().numpy()
            for op_idx, op_name in enumerate(self.selection_operator_names):
                prob_value = probs_cpu[op_idx]
                is_selected = "✓" if op_name in selected_names_layers[layer_idx] else ""
                logger.info(f"{op_name:<25} {prob_value:>6.4f} ({prob_value*100:>5.2f}%)  {is_selected:<10}")

        logger.info(f"{'='*80}\n")

    async def _dispatch_operator(self, task, dep_solution, task_outputs, problem):
        """GSM8K-specific operator dispatch for DAG execution."""
        import time
        op_name = task.operator_name
        selected_operator = self.selection_operator_instances[op_name]
        task_start = time.time()

        if op_name in ["Generate", "GenerateCoT"]:
            result = await selected_operator(input=problem, instruction=self.prompt_module.MATH_SOLVE_PROMPT, return_usage=True)
            latency = time.time() - task_start
            cp_token_count = result.get('_usage_tokens', 0)
            return result, {"type": "generate", "solution": result.get('response', ""), "operator": op_name, "latency": latency, "iterations": 1, "cp_token_count": cp_token_count, "_raw_prompt_tokens": result.get('_usage_prompt_tokens', 0), "_raw_completion_tokens": result.get('_usage_tokens', 0)}
        elif op_name == "MultiGenerateCoT":
            result = await selected_operator(input=problem, instruction=self.prompt_module.MATH_SOLVE_PROMPT, return_usage=True)
            latency = time.time() - task_start
            raw_tokens = result.get('_usage_tokens', 0)
            if isinstance(result, dict) and 'response' in result:
                num_iterations = len(result['response']) if isinstance(result['response'], list) else 1
                cp_token_count = raw_tokens / num_iterations if num_iterations > 0 else raw_tokens
                return result, {"type": "multi_generate", "solutions": [res.get('response', "") for res in result['response']], "operator": op_name, "latency": latency, "iterations": num_iterations, "cp_token_count": cp_token_count, "_raw_prompt_tokens": result.get('_usage_prompt_tokens', 0), "_raw_completion_tokens": raw_tokens}
            else:
                return result, {"type": "multi_generate", "solutions": [], "operator": op_name, "latency": latency, "iterations": 0, "cp_token_count": raw_tokens, "_raw_prompt_tokens": 0, "_raw_completion_tokens": raw_tokens}
        elif op_name == "SelfRefine":
            result = await selected_operator(problem=problem, solution=dep_solution, return_usage=True)
            latency = time.time() - task_start
            cp_token_count = result.get('_usage_tokens', 0)
            return result, {"type": "refine", "solution": result.get('response', ""), "operator": op_name, "latency": latency, "iterations": 1, "cp_token_count": cp_token_count, "_raw_prompt_tokens": result.get('_usage_prompt_tokens', 0), "_raw_completion_tokens": result.get('_usage_tokens', 0)}
        elif op_name == "Programmer":
            result = await selected_operator(problem=problem, analysis=dep_solution, return_usage=True)
            refined_solution = await self.custom(input=problem + f"\nCode output: {result['code']}", instruction=self.prompt_module.REFINE_ANSWER_PROMPT, return_usage=True)
            latency = time.time() - task_start
            llm_tokens = result.get('_usage_tokens', 0) + refined_solution.get('_usage_tokens', 0)
            tool_exec_time = result.get('tool_exec_time', 0.0)
            cp_token_count = llm_tokens + tool_exec_time * 50.0
            return {"code": result.get("code", ""), "output": result.get("output", ""), "refined": refined_solution.get("response", "")}, {"type": "programmer", "solution": refined_solution['response'], "operator": op_name, "latency": latency, "iterations": 1, "cp_token_count": cp_token_count, "_raw_prompt_tokens": result.get('_usage_prompt_tokens', 0) + refined_solution.get('_usage_prompt_tokens', 0), "_raw_completion_tokens": result.get('_usage_tokens', 0) + refined_solution.get('_usage_tokens', 0)}
        elif op_name == "ScEnsemble":
            resolved_solutions = []
            if task.depends_on_solutions:
                for dep_id in task.depends_on_solutions:
                    if dep_id in task_outputs:
                        dep = task_outputs[dep_id]
                        if dep.get("type") == "multi_generate":
                            resolved_solutions.extend(dep.get("solutions", []))
                        else:
                            resolved_solutions.append(dep.get("solution", ""))
            if not resolved_solutions:
                latency = time.time() - task_start
                return "skipped (no solutions)", {"type": "noop", "solution": dep_solution, "operator": op_name, "latency": latency, "iterations": 0, "cp_token_count": 0, "_raw_prompt_tokens": 0, "_raw_completion_tokens": 0}
            else:
                result = await selected_operator(problem=problem, solutions=resolved_solutions, return_usage=True)
                latency = time.time() - task_start
                cp_token_count = result.get('_usage_tokens', 0)
                return {"selected": result.get('response', ""), "num_solutions": len(resolved_solutions)}, {"type": "ensemble", "solution": result.get('response', ""), "operator": op_name, "latency": latency, "iterations": 1, "cp_token_count": cp_token_count, "_raw_prompt_tokens": result.get('_usage_prompt_tokens', 0), "_raw_completion_tokens": result.get('_usage_tokens', 0)}
        else:
            latency = time.time() - task_start
            return "noop", {"type": "noop", "solution": dep_solution, "operator": op_name, "latency": latency, "iterations": 0, "cp_token_count": 0, "_raw_prompt_tokens": 0, "_raw_completion_tokens": 0}

    async def __call__(self, problem: str, log_path: str = None):
        import time

        # Lazily init tracer on first call
        if self.tracer is None and log_path:
            self.tracer = OperatorTracer(log_dir=log_path, dataset=self.dataset)

        # Track bottleneck timings for this problem
        problem_start_time = time.time()

        # Run controller to select operators for all layers (single forward pass)
        controller_start = time.time()
        log_probs_layers, selected_names_layers, probs_layers = self.controller.forward(problem, self.operator_embeddings, self.selection_operator_names)
        controller_time = time.time() - controller_start

        # Store probabilities for aggregation (will be used by benchmark)
        self.last_probs_layers = probs_layers

        # Log probability distributions for each layer
        if log_path:
            self._log_operator_probabilities(probs_layers, selected_names_layers, log_path, problem)

        # Decompose layer log probs into per-operator log probs for critical path tracking
        log_probs_per_layer = []  # List[torch.Tensor] - individual operator log probs
        operator_names_per_layer = []  # List[List[str]]

        for layer_idx, (selected_names, probs) in enumerate(zip(selected_names_layers, probs_layers)):
            if not selected_names:
                log_probs_per_layer.append(torch.tensor([], device=self.device))
                operator_names_per_layer.append([])
                continue

            # Extract individual log probs from the probability distribution
            log_probs_full = torch.log(probs + 1e-10)
            selected_indices = [self.selection_operator_names.index(name) for name in selected_names]
            individual_log_probs = log_probs_full[selected_indices]

            log_probs_per_layer.append(individual_log_probs)
            operator_names_per_layer.append(selected_names)

        current_solution = ""
        solutions = []
        sum_log_prob = 0.0

        # Accumulators for per-problem cost computation
        problem_prompt_tokens = 0
        problem_completion_tokens = 0

        # Track per-operator latency and iteration counts
        operator_latencies = {}  # {operator_name: [latencies]}
        operator_iterations = {}  # {operator_name: [iteration_counts]}

        # Track per-layer operator latencies for critical path tracking
        operator_latencies_per_layer = []

        # Track per-layer operator token counts for virtual token calculation
        operator_token_counts_per_layer = []

        # Track bottleneck breakdown
        total_operator_time = 0.0

        # --- Scheduler path (when train_scheduler enabled) ---
        if self.use_scheduler:
            from maas.ext.maas.scripts.optimized.scheduler import schedule
            from maas.ext.maas.scripts.optimized.dag_executor import (
                execute_dag, compute_critical_path, collect_metrics,
                reconstruct_solutions, rebuild_log_probs,
            )
            plan = schedule(selected_names_layers)

            async def dispatch_fn(task, dep_solution, task_outputs):
                return await self._dispatch_operator(task, dep_solution, task_outputs, problem)

            task_outputs, total_operator_time, gate_log_probs, _, _gate_state = await execute_dag(
                plan=plan, problem=problem, dispatch_fn=dispatch_fn,
                tracer=self.tracer, device=self.device,
                pruning_gate=None,
                problem_id=problem[:100], dataset=self.dataset,
            )

            metrics = collect_metrics(plan, task_outputs, self.selection_operator_names)
            solutions, current_solution = reconstruct_solutions(plan, task_outputs)
            sched_log_probs = rebuild_log_probs(plan, probs_layers, self.selection_operator_names, self.device)

            sum_log_prob = sum(log_probs_layers)
            problem_prompt_tokens = metrics["problem_prompt_tokens"]
            problem_completion_tokens = metrics["problem_completion_tokens"]
            operator_latencies = metrics["operator_latencies"]
            operator_iterations = metrics["operator_iterations"]
            operator_names_per_layer = metrics["operator_names_per_layer"]
            operator_latencies_per_layer = metrics["operator_latencies_per_layer"]
            operator_token_counts_per_layer = metrics["operator_token_counts_per_layer"]
            log_probs_per_layer = sched_log_probs

            pass  # Skip layer-by-layer; fall through to post-processing

        for layer_idx, selected_names in enumerate(selected_names_layers if not self.use_scheduler else []):
            # All operators in the same layer are independent and can run in parallel
            # They all receive the same state from previous layers
            if not selected_names:
                continue

            if self.parallel_execution:
                # PARALLEL MODE: Run all operators in the layer in parallel
                async def execute_operator(op_name):
                    """Execute a single operator with the current state and track latency and tokens"""
                    import time
                    selected_operator = self.selection_operator_instances[op_name]
                    start_time = time.time()

                    def _trace(op_name, raw_result, ret, layer_idx):
                        if self.tracer:
                            self.tracer.log(layer=layer_idx, operator=op_name, input=problem, output=raw_result, latency=ret.get("latency", 0), tokens=ret.get("cp_token_count", 0))
                        return ret

                    if op_name in ["Generate", "GenerateCoT"]:
                        result = await selected_operator(input=problem, instruction=self.prompt_module.MATH_SOLVE_PROMPT, return_usage=True)
                        latency = time.time() - start_time
                        cp_token_count = result.get('_usage_tokens', 0)
                        return _trace(op_name, result, {"type": "generate", "solution": result.get('response', ""), "operator": op_name, "latency": latency, "iterations": 1, "cp_token_count": cp_token_count, "_raw_prompt_tokens": result.get('_usage_prompt_tokens', 0), "_raw_completion_tokens": result.get('_usage_tokens', 0)}, layer_idx)
                    elif op_name == "MultiGenerateCoT":
                        result = await selected_operator(input=problem, instruction=self.prompt_module.MATH_SOLVE_PROMPT, return_usage=True)
                        latency = time.time() - start_time
                        raw_tokens = result.get('_usage_tokens', 0)
                        if isinstance(result, dict) and 'response' in result:
                            num_iterations = len(result['response']) if isinstance(result['response'], list) else 1
                            cp_token_count = raw_tokens / num_iterations if num_iterations > 0 else raw_tokens
                            return _trace(op_name, result, {"type": "multi_generate", "solutions": [res.get('response', "") for res in result['response']], "operator": op_name, "latency": latency, "iterations": num_iterations, "cp_token_count": cp_token_count, "_raw_prompt_tokens": result.get('_usage_prompt_tokens', 0), "_raw_completion_tokens": raw_tokens}, layer_idx)
                        else:
                            logger.error(f"Expected dict with 'response' from MultiGenerateCoT, got {type(result)}")
                            return _trace(op_name, result, {"type": "multi_generate", "solutions": [], "operator": op_name, "latency": latency, "iterations": 0, "cp_token_count": raw_tokens, "_raw_prompt_tokens": result.get('_usage_prompt_tokens', 0) if isinstance(result, dict) else 0, "_raw_completion_tokens": raw_tokens}, layer_idx)
                    elif op_name == "SelfRefine":
                        result = await selected_operator(problem=problem, solution=current_solution, return_usage=True)
                        latency = time.time() - start_time
                        cp_token_count = result.get('_usage_tokens', 0)
                        return _trace(op_name, result, {"type": "refine", "solution": result.get('response', ""), "operator": op_name, "latency": latency, "iterations": 1, "cp_token_count": cp_token_count, "_raw_prompt_tokens": result.get('_usage_prompt_tokens', 0), "_raw_completion_tokens": result.get('_usage_tokens', 0)}, layer_idx)
                    elif op_name == "Programmer":
                        result = await selected_operator(problem=problem, analysis=current_solution, return_usage=True)
                        refined_solution = await self.custom(input=problem + f"\nCode output: {result['code']}", instruction=self.prompt_module.REFINE_ANSWER_PROMPT, return_usage=True)
                        latency = time.time() - start_time
                        llm_tokens = result.get('_usage_tokens', 0) + refined_solution.get('_usage_tokens', 0)
                        tool_exec_time = result.get('tool_exec_time', 0.0)
                        cp_token_count = llm_tokens + tool_exec_time * 50.0
                        return _trace(op_name, {"code": result.get("code", ""), "output": result.get("output", ""), "refined": refined_solution.get("response", "")}, {"type": "programmer", "solution": refined_solution['response'], "operator": op_name, "latency": latency, "iterations": 1, "cp_token_count": cp_token_count, "_raw_prompt_tokens": result.get('_usage_prompt_tokens', 0) + refined_solution.get('_usage_prompt_tokens', 0), "_raw_completion_tokens": result.get('_usage_tokens', 0) + refined_solution.get('_usage_tokens', 0)}, layer_idx)
                    elif op_name == "ScEnsemble":
                        if not solutions:
                            latency = time.time() - start_time
                            return _trace(op_name, "skipped (no solutions)", {"type": "noop", "solution": current_solution, "operator": op_name, "latency": latency, "iterations": 0, "cp_token_count": 0, "_raw_prompt_tokens": 0, "_raw_completion_tokens": 0}, layer_idx)
                        result = await selected_operator(problem=problem, solutions=solutions, return_usage=True)
                        latency = time.time() - start_time
                        cp_token_count = result.get('_usage_tokens', 0)
                        return _trace(op_name, {"selected": result.get('response', ""), "num_solutions": len(solutions)}, {"type": "ensemble", "solution": result.get('response', ""), "operator": op_name, "latency": latency, "iterations": 1, "cp_token_count": cp_token_count, "_raw_prompt_tokens": result.get('_usage_prompt_tokens', 0), "_raw_completion_tokens": result.get('_usage_tokens', 0)}, layer_idx)
                    else:
                        latency = time.time() - start_time
                        cp_token_count = 0
                        return _trace(op_name, "noop", {"type": "noop", "solution": current_solution, "operator": op_name, "latency": latency, "iterations": 0, "cp_token_count": cp_token_count, "_raw_prompt_tokens": 0, "_raw_completion_tokens": 0}, layer_idx)

                # Run all operators in the layer in parallel
                results = await asyncio.gather(*[execute_operator(op_name) for op_name in selected_names])

                # Track latencies and token counts for this layer in the same order as selected_names
                layer_latencies = []
                layer_token_counts = []
                for op_name in selected_names:
                    # Find the result for this operator
                    op_result = next((r for r in results if r.get("operator") == op_name), None)
                    if op_result:
                        layer_latencies.append(op_result.get("latency", 0.0))
                        layer_token_counts.append(op_result.get("cp_token_count", 0))
                    else:
                        layer_latencies.append(0.0)
                        layer_token_counts.append(0)
                operator_latencies_per_layer.append(layer_latencies)
                operator_token_counts_per_layer.append(layer_token_counts)

                # Process results and update state
                for result in results:
                    # Track latency for each operator
                    op_name = result.get("operator", "Unknown")
                    op_latency = result.get("latency", 0.0)
                    op_iterations = result.get("iterations", 1)

                    # Accumulate tokens for per-problem cost
                    problem_prompt_tokens += result.get("_raw_prompt_tokens", 0)
                    problem_completion_tokens += result.get("_raw_completion_tokens", 0)

                    if op_name not in operator_latencies:
                        operator_latencies[op_name] = []
                        operator_iterations[op_name] = []
                    operator_latencies[op_name].append(op_latency)
                    operator_iterations[op_name].append(op_iterations)

                    if result["type"] == "generate":
                        solutions.append(result["solution"])
                        current_solution = result["solution"]
                    elif result["type"] == "multi_generate":
                        solutions.extend(result["solutions"])
                        if result["solutions"]:
                            current_solution = result["solutions"][-1]
                    elif result["type"] == "refine":
                        solutions.append(result["solution"])
                        current_solution = result["solution"]
                    elif result["type"] == "programmer":
                        solutions.append(result["solution"])
                        current_solution = result["solution"]
                    elif result["type"] == "ensemble":
                        solutions = [result["solution"]]
                        current_solution = result["solution"]

                # Track total operator execution time (max time since operators run in parallel within layer)
                if len(results) > 0:
                    layer_time = max(result.get("latency", 0.0) for result in results)
                    total_operator_time += layer_time
            else:
                # SEQUENTIAL MODE: Run operators one by one (original behavior)
                layer_start_time = time.time()
                layer_latencies = []
                layer_token_counts = []

                for op_name in selected_names:
                    selected_operator = self.selection_operator_instances[op_name]
                    op_start_time = time.time()

                    if op_name in ["Generate", "GenerateCoT"]:
                        result = await selected_operator(input=problem, instruction=self.prompt_module.MATH_SOLVE_PROMPT, return_usage=True)
                        new_solution = result.get('response', "")
                        solutions.append(new_solution)
                        current_solution = new_solution
                        if self.tracer:
                            self.tracer.log(layer=layer_idx, operator=op_name, input=problem, output=result, latency=time.time()-op_start_time, tokens=result.get('_usage_tokens', 0))
                    elif op_name == "SelfRefine":
                        result = await selected_operator(problem=problem, solution=current_solution, return_usage=True)
                        new_solution = result.get('response', "")
                        solutions.append(new_solution)
                        current_solution = new_solution
                        if self.tracer:
                            self.tracer.log(layer=layer_idx, operator=op_name, input=problem, output=result, latency=time.time()-op_start_time, tokens=result.get('_usage_tokens', 0))
                    elif op_name == "Programmer":
                        result = await selected_operator(problem=problem, analysis=current_solution, return_usage=True)
                        refined_solution = await self.custom(input=problem + f"\nCode output: {result['code']}", instruction=self.prompt_module.REFINE_ANSWER_PROMPT, return_usage=True)
                        new_solution = refined_solution['response']
                        solutions.append(new_solution)
                        current_solution = new_solution
                        if self.tracer:
                            self.tracer.log(layer=layer_idx, operator=op_name, input=problem, output={"code": result.get("code", ""), "output": result.get("output", ""), "refined": new_solution}, latency=time.time()-op_start_time, tokens=result.get('_usage_tokens', 0) + refined_solution.get('_usage_tokens', 0))
                    elif op_name == "ScEnsemble":
                        result = await selected_operator(problem=problem, solutions=solutions, return_usage=True)
                        solutions = []
                        new_solution = result.get('response', "")
                        solutions.append(new_solution)
                        current_solution = new_solution
                        if self.tracer:
                            self.tracer.log(layer=layer_idx, operator=op_name, input=problem, output={"selected": new_solution, "num_solutions": len(solutions)}, latency=time.time()-op_start_time, tokens=result.get('_usage_tokens', 0))
                    elif op_name == "MultiGenerateCoT":
                        result = await selected_operator(input=problem, instruction=self.prompt_module.MATH_SOLVE_PROMPT, return_usage=True)
                        if isinstance(result, dict) and 'response' in result:
                            num_iterations = len(result['response']) if isinstance(result['response'], list) else 1
                            for res in result['response']:
                                new_solution = res.get('response', "")
                                solutions.append(new_solution)
                            current_solution = new_solution
                        else:
                            logger.error(f"Expected dict with 'response' from MultiGenerateCoT, got {type(result)}")
                            new_solution = current_solution
                        if self.tracer:
                            self.tracer.log(layer=layer_idx, operator=op_name, input=problem, output=result, latency=time.time()-op_start_time, tokens=result.get('_usage_tokens', 0) if isinstance(result, dict) else 0)
                    else:
                        new_solution = current_solution
                        if self.tracer:
                            self.tracer.log(layer=layer_idx, operator=op_name, input=problem, output="noop", latency=0, tokens=0)

                    # Track latency and tokens for this operator
                    op_latency = time.time() - op_start_time
                    raw_tokens = result.get('_usage_tokens', 0) if isinstance(result, dict) else 0

                    # Accumulate tokens for per-problem cost
                    problem_prompt_tokens += result.get('_usage_prompt_tokens', 0) if isinstance(result, dict) else 0
                    problem_completion_tokens += raw_tokens

                    # For Programmer operator, add refined_solution tokens too
                    if op_name == "Programmer":
                        problem_prompt_tokens += refined_solution.get('_usage_prompt_tokens', 0)
                        problem_completion_tokens += refined_solution.get('_usage_tokens', 0)
                        raw_tokens += refined_solution.get('_usage_tokens', 0)

                    op_iterations = 1
                    if op_name == "MultiGenerateCoT" and isinstance(result, dict) and 'response' in result:
                        op_iterations = len(result['response']) if isinstance(result['response'], list) else 1

                    # For MultiGenerateCoT, divide tokens by iterations to get effective tokens
                    cp_token_count = raw_tokens / op_iterations if op_iterations > 0 else raw_tokens

                    if op_name not in operator_latencies:
                        operator_latencies[op_name] = []
                        operator_iterations[op_name] = []
                    operator_latencies[op_name].append(op_latency)
                    operator_iterations[op_name].append(op_iterations)

                    # Track latency and tokens for this layer
                    layer_latencies.append(op_latency)
                    layer_token_counts.append(cp_token_count)

                # Store layer latencies and token counts for critical path tracking
                operator_latencies_per_layer.append(layer_latencies)
                operator_token_counts_per_layer.append(layer_token_counts)

                # Track total operator execution time (sum time since operators run sequentially)
                layer_time = time.time() - layer_start_time
                total_operator_time += layer_time

            sum_log_prob += log_probs_layers[layer_idx].item()

        # Final ensemble if multiple solutions (post-processing)
        ensemble_tokens = 0.0
        ensemble_prompt_tokens = 0
        post_process_start = time.time()
        if len(solutions) > 1:
            final_solution = await self.sc_ensemble(solutions=solutions, problem=problem, return_usage=True)
            ensemble_tokens = final_solution.get('_usage_tokens', 0)
            ensemble_prompt_tokens = final_solution.get('_usage_prompt_tokens', 0)
            problem_prompt_tokens += ensemble_prompt_tokens
            problem_completion_tokens += ensemble_tokens
            final_solution = final_solution['response']
            if self.tracer:
                self.tracer.log(layer=-1, operator="ScEnsemble_post", input=problem, output={"selected": final_solution, "num_solutions": len(solutions)}, latency=time.time()-post_process_start, tokens=ensemble_tokens)
        else:
            final_solution = current_solution
        ensemble_time = time.time() - post_process_start

        # GSM8K-specific: Verify solution with programmer (post-processing)
        programmer_tokens = 0.0
        programmer_time = 0.0
        verification = None
        if not getattr(self, 'skip_postprocess', False):
            programmer_start = time.time()
            verification = await self.programmer(problem=problem, analysis=final_solution, return_usage=True)
            llm_tokens = verification.get('_usage_tokens', 0)
            tool_exec_time = verification.get('tool_exec_time', 0.0)
            programmer_tokens = llm_tokens + tool_exec_time * 50.0

            # Accumulate tokens for per-problem cost
            problem_prompt_tokens += verification.get('_usage_prompt_tokens', 0)
            problem_completion_tokens += llm_tokens
            programmer_time = time.time() - programmer_start
            if self.tracer:
                self.tracer.log(layer=-2, operator="Programmer_post", input=problem, output={"code": verification.get("code", ""), "output": verification.get("output", "")}, latency=programmer_time, tokens=int(programmer_tokens))

        # Append post-processing as additional pseudo-layers for per-operator tracking
        post_ops_names = []
        post_ops_tokens = []
        post_ops_latencies = []
        if ensemble_tokens > 0:
            post_ops_names.append("ScEnsemble_post")
            post_ops_tokens.append(ensemble_tokens)
            post_ops_latencies.append(ensemble_time)
        if programmer_tokens > 0:
            post_ops_names.append("Programmer_post")
            post_ops_tokens.append(programmer_tokens)
            post_ops_latencies.append(programmer_time)
        if post_ops_names:
            operator_names_per_layer.append(post_ops_names)
            operator_token_counts_per_layer.append(post_ops_tokens)
            operator_latencies_per_layer.append(post_ops_latencies)

        # Calculate total problem latency and breakdown AFTER all work is done
        total_problem_time = time.time() - problem_start_time

        # Store operator latency statistics in graph for later aggregation
        self.last_operator_latencies = operator_latencies
        self.last_operator_iterations = operator_iterations

        # Store bottleneck breakdown for this problem
        self.last_bottleneck_breakdown = {
            'controller_time': controller_time,
            'operator_time': total_operator_time,
            'total_time': total_problem_time,
            'overhead_time': total_problem_time - controller_time - total_operator_time
        }

        # Calculate total virtual tokens across all layers
        total_virtual_tokens = 0.0
        for layer_tokens in operator_token_counts_per_layer:
            if layer_tokens:
                if self.parallel_execution:
                    # In parallel mode, count max tokens per layer (critical path)
                    total_virtual_tokens += max(layer_tokens)
                else:
                    # In sequential mode, sum all tokens in layer
                    total_virtual_tokens += sum(layer_tokens)

        # Add post-processing tokens (ScEnsemble and Programmer verification)
        total_virtual_tokens += ensemble_tokens
        total_virtual_tokens += programmer_tokens

        # Build layer_operator_info dict for critical path tracking
        layer_operator_info = {
            'log_probs_per_layer': log_probs_per_layer,
            'operator_names_per_layer': operator_names_per_layer,
            'operator_latencies_per_layer': operator_latencies_per_layer,
            'operator_token_counts_per_layer': operator_token_counts_per_layer,  # NEW: for virtual token calculation
        }

        # Compute per-problem cost from accumulated tokens
        model_name = self.llm.config.model if hasattr(self.llm, 'config') and hasattr(self.llm.config, 'model') else ""
        token_prices = TOKEN_COSTS.get(model_name, {"prompt": 0.005, "completion": 0.015})  # default to gpt-4o prices
        problem_cost = (problem_prompt_tokens * token_prices["prompt"] + problem_completion_tokens * token_prices["completion"]) / 1000

        if verification and verification['output']:
            return verification['output'], problem_cost, sum_log_prob, total_virtual_tokens, layer_operator_info
        else:
            return final_solution, problem_cost, sum_log_prob, total_virtual_tokens, layer_operator_info

# Keep backward compatibility
Workflow = GSM8KGraph
