"""MaaS optimizer.

Trains the Multi-Layer Controller via REINFORCE under a Lagrangian-constrained
objective, then runs Phase 1.5 (trace collection) -> Phase 2 (gate training)
-> Phase 3 (test with gate). Critical-path-aware credit assignment, parallel
DAG execution and reward normalization are always on.
"""

import asyncio
import glob
import os
import time
from typing import List, Literal

import torch

from maas.ext.maas.scripts.evaluator import DatasetType
from maas.ext.maas.scripts.optimizer_utils.data_utils import DataUtils
from maas.ext.maas.scripts.optimizer_utils.evaluation_utils import EvaluationUtils
from maas.ext.maas.scripts.optimizer_utils.graph_utils import GraphUtils
from maas.logs import logger
from maas.ext.maas.models.utils import get_sentence_embedding
from maas.ext.maas.models.controller import MultiLayerController

QuestionType = Literal["math", "qa"]
OptimizerType = Literal["Graph", "Test"]


class Optimizer:
    def __init__(
        self,
        dataset: DatasetType,
        question_type: QuestionType,
        opt_llm_config,
        exec_llm_config,
        operators: List,
        sample: int,
        optimized_path: str,
        round: int = 1,
        batch_size: int = 4,
        lr: float = 0.01,
        latency_weight: float = 0.005,
        cost_weight: float = 3.0,
        threshold: float = 0.3,
        num_layers: int = 4,
        max_graph_retries: int = 3,
        max_concurrent_tasks: int = 1,
        acc_baseline: float = 0.0,
        lagrangian_lr: float = 0.1,
        pruning_gate=None,
        checkpoint_path: str = None,
        code_path: str = None,
    ) -> None:
        self.dataset = dataset
        self.type = question_type
        self.optimize_llm_config = opt_llm_config
        self.execute_llm_config = exec_llm_config
        self.operators = operators
        self.sample = sample
        self.round = round
        self.batch_size = batch_size
        self.lr = lr
        self.latency_weight = latency_weight
        self.cost_weight = cost_weight
        self.threshold = threshold
        self.num_layers = num_layers
        self.max_graph_retries = max_graph_retries
        self.max_concurrent_tasks = max_concurrent_tasks
        self.acc_baseline = acc_baseline
        self.lagrangian_lr = lagrangian_lr
        self.pruning_gate = pruning_gate
        self.checkpoint_path = checkpoint_path

        self.root_path = f"{optimized_path}/{self.dataset}"
        self.code_path = f"{code_path}/{self.dataset}" if code_path else self.root_path
        self.graph = None

        self.graph_utils = GraphUtils(self.root_path, self.code_path)
        self.data_utils = DataUtils(self.root_path)
        self.evaluation_utils = EvaluationUtils(self.root_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.controller = MultiLayerController(
            device=self.device,
            num_layers=self.num_layers,
            threshold=self.threshold,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.controller.parameters(), lr=self.lr)

    def optimize(self, mode: OptimizerType = "Graph"):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        if mode == "Test":
            return loop.run_until_complete(self.test())
        try:
            score = loop.run_until_complete(self._optimize_graph_maas())
        except Exception as e:
            import traceback
            logger.info(f"Error: {e}\n{traceback.format_exc()}")
            score = None
        logger.info(f"Score: {score}")
        time.sleep(5)

    async def _optimize_graph_maas(self):
        graph_path = f"{self.root_path}/train"
        data = self.data_utils.load_results(graph_path)

        operator_descriptions = self.graph_utils.load_operators_description_maas(self.operators)
        precomputed_operator_embeddings = torch.stack(
            [get_sentence_embedding(d) for d in operator_descriptions]
        ).to(self.device)
        directory = self.graph_utils.create_round_directory(graph_path, self.round)
        self.graph = self.graph_utils.load_graph_maas(graph_path)

        params = self._build_params(precomputed_operator_embeddings)
        avg_score = await self.evaluation_utils.evaluate_graph_maas(
            self, directory, data, initial=False, params=params
        )
        return avg_score

    def _build_params(self, operator_embeddings):
        return {
            "operator_embeddings": operator_embeddings,
            "controller": self.controller,
            "execute_llm_config": self.execute_llm_config,
            "dataset": self.dataset,
            "optimizer": self.optimizer,
            "sample": self.sample,
            "latency_weight": self.latency_weight,
            "cost_weight": self.cost_weight,
            "num_layers": self.num_layers,
            "threshold": self.threshold,
            "max_graph_retries": self.max_graph_retries,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "acc_baseline": self.acc_baseline,
            "lagrangian_lr": self.lagrangian_lr,
            "pruning_gate": self.pruning_gate,
        }

    def _load_controller_checkpoint(self, checkpoint_path_override=None):
        """Find the latest checkpoint matching the current configuration and load it."""
        if checkpoint_path_override:
            logger.info(f"Loading checkpoint (override): {checkpoint_path_override}")
            checkpoint = torch.load(checkpoint_path_override, map_location=self.device, weights_only=True)
            self.controller.load_state_dict(checkpoint)
            self.controller.eval()
            return

        pth_dir = self.graph_utils.create_round_directory(f"{self.root_path}/train", self.round)
        base_name = f"{self.dataset}_controller_sample{self.sample}"
        suffix = (
            f"_lat{self.latency_weight:.4f}".replace(".", "_")
            + f"_th{self.threshold:.2f}".replace(".", "_")
            + f"_cw{self.cost_weight:.1f}".replace(".", "_")
            + f"_nl{self.num_layers}"
            + f"_lag{self.lagrangian_lr:.3f}".replace(".", "_")
        )
        prefix = os.path.join(pth_dir, f"{base_name}{suffix}")
        candidates = sorted(glob.glob(prefix + "_[0-9]*.pth"))
        controller_path = candidates[-1] if candidates else None

        if controller_path is None:
            # Fall back to any matching base_name file (less strict)
            candidates = sorted(glob.glob(os.path.join(pth_dir, f"{base_name}*.pth")))
            if candidates:
                controller_path = candidates[-1]

        if controller_path and os.path.exists(controller_path):
            logger.info(f"Loading checkpoint: {controller_path}")
            checkpoint = torch.load(controller_path, map_location=self.device, weights_only=True)
            self.controller.load_state_dict(checkpoint)
            self.controller.eval()
        else:
            raise FileNotFoundError(f"Controller checkpoint not found in {pth_dir}")

    async def collect_traces(self):
        """Run the converged controller on the train set to collect DAG traces.

        Test graph is loaded (it has the scheduler path), but the train data
        file is used. No gradient updates. Output traces go to the train
        directory so train_pruning_gate.py can pick them up.
        """
        self._load_controller_checkpoint(self.checkpoint_path)

        # Test graph (has scheduler), but on train data
        graph_path = f"{self.root_path}/test"
        self.graph = self.graph_utils.load_graph_maas(graph_path)

        operator_descriptions = self.graph_utils.load_operators_description_maas(self.operators)
        precomputed_operator_embeddings = torch.stack(
            [get_sentence_embedding(d) for d in operator_descriptions]
        ).to(self.device)

        train_path = f"{self.root_path}/train"
        directory = self.graph_utils.create_round_directory(train_path, self.round)
        train_data_path = f"maas/ext/maas/data/{self.dataset.lower()}_train.jsonl"
        logger.info(f"Collecting traces to: {directory}")

        params = self._build_params(precomputed_operator_embeddings)
        params["data_path_override"] = train_data_path

        score = await self.evaluation_utils.evaluate_graph_test_maas(
            self, directory, is_test=True, params=params
        )
        logger.info(f"Trace collection complete. Train-set score: {score}")
        return score

    async def test(self):
        """Evaluate the controller on the test set."""
        graph_path = f"{self.root_path}/test"
        self.graph = self.graph_utils.load_graph_maas(graph_path)
        directory = self.graph_utils.create_round_directory(graph_path, self.round)
        operator_descriptions = self.graph_utils.load_operators_description_maas(self.operators)
        precomputed_operator_embeddings = torch.stack(
            [get_sentence_embedding(d) for d in operator_descriptions]
        ).to(self.device)

        self._load_controller_checkpoint(self.checkpoint_path)

        json_file_path = self.data_utils.get_results_file_path(graph_path)
        data = self.data_utils.load_results(graph_path)
        params = self._build_params(precomputed_operator_embeddings)

        score = await self.evaluation_utils.evaluate_graph_test_maas(
            self, directory, is_test=True, params=params
        )
        new_data = self.data_utils.create_result_data(self.round, score)
        data.append(new_data)
        self.data_utils.save_results(json_file_path, data)
        return score
