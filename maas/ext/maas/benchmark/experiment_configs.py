from typing import Dict, List


class ExperimentConfig:
    def __init__(self, dataset: str, question_type: str, operators: List[str],
                 acc_baseline: float):
        self.dataset = dataset
        self.question_type = question_type
        self.operators = operators
        self.acc_baseline = acc_baseline


_OPERATORS = ["Generate", "GenerateCoT", "MultiGenerateCoT", "ScEnsemble",
              "Programmer", "SelfRefine", "EarlyStop"]

EXPERIMENT_CONFIGS: Dict[str, ExperimentConfig] = {
    "MATH": ExperimentConfig(
        dataset="MATH", question_type="math", operators=_OPERATORS,
        acc_baseline=0.50,
    ),
    "GSM8K": ExperimentConfig(
        dataset="GSM8K", question_type="math", operators=_OPERATORS,
        acc_baseline=0.85,
    ),
    "MMLU_Pro": ExperimentConfig(
        dataset="MMLU_Pro", question_type="qa", operators=_OPERATORS,
        acc_baseline=0.65,
    ),
}
