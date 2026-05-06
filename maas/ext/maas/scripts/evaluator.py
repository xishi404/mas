from typing import Dict, Literal

from maas.ext.maas.benchmark.benchmark import BaseBenchmark
from maas.ext.maas.benchmark.gsm8k import GSM8KBenchmark
from maas.ext.maas.benchmark.math import MATHBenchmark
from maas.ext.maas.benchmark.mmlu_pro import MMLUProBenchmark

DatasetType = Literal["GSM8K", "MATH", "MMLU_Pro"]


class Evaluator:
    def __init__(self, eval_path: str, batch_size: int):
        self.eval_path = eval_path
        self.batch_size = batch_size
        self.dataset_configs: Dict[DatasetType, BaseBenchmark] = {
            "GSM8K": GSM8KBenchmark,
            "MATH": MATHBenchmark,
            "MMLU_Pro": MMLUProBenchmark,
        }

    async def graph_evaluate(
        self, dataset: DatasetType, graph, params: dict, path: str, is_test: bool = False
    ):
        if dataset not in self.dataset_configs:
            raise ValueError(f"Unsupported dataset: {dataset}")

        data_path = params.get("data_path_override") or self._get_data_path(dataset, is_test)
        benchmark_class = self.dataset_configs[dataset]

        benchmark = benchmark_class(
            name=dataset,
            file_path=data_path,
            log_path=path,
            batch_size=self.batch_size,
            controller=params["controller"],
            operator_embeddings=params["operator_embeddings"],
            optimizer=params["optimizer"],
            latency_weight=params.get("latency_weight", 0.005),
            cost_weight=params.get("cost_weight", 3.0),
            num_layers=params.get("num_layers", 4),
            threshold=params.get("threshold", 0.3),
            max_graph_retries=params.get("max_graph_retries", 3),
            max_concurrent_tasks=params.get("max_concurrent_tasks", 1),
            acc_baseline=params.get("acc_baseline", 0.0),
            lagrangian_lr=params.get("lagrangian_lr", 0.1),
            pruning_gate=params.get("pruning_gate", None),
        )
        configured_graph = await self._configure_graph(dataset, graph, params)
        score = await benchmark.run_evaluation(configured_graph, None, is_test, params["sample"])
        return score, benchmark

    async def _configure_graph(self, dataset, graph, params: dict):
        controller = params.get("controller")
        operator_embeddings = params.get("operator_embeddings")
        llm_config = params.get("execute_llm_config")
        dataset_config = params.get("dataset")
        pruning_gate = params.get("pruning_gate", None)
        configured_graph = graph(
            name=dataset,
            llm_config=llm_config,
            dataset=dataset_config,
            controller=controller,
            operator_embeddings=operator_embeddings,
            pruning_gate=pruning_gate,
        )
        return configured_graph

    def _get_data_path(self, dataset: DatasetType, test: bool) -> str:
        base_path = f"maas/ext/maas/data/{dataset.lower()}"
        return f"{base_path}_test.jsonl" if test else f"{base_path}_train.jsonl"
