# MaaS — Multi-Agent System with Learned Pruning Gate

Critical-path-aware Lagrangian optimization of LLM-based multi-agent workflows,
with an online learned pruning gate for early stopping during DAG execution.

## Reproducing Experiments

### Prerequisites

```bash
pip install -r requirements.txt
```

Configure API keys in `config/config2.yaml` (copy from `config/config2.example.yaml`).

### Dataset Setup

Place the GSM8K, MATH and MMLU_Pro datasets under `maas/ext/maas/data/`:

```
maas/ext/maas/data/
├── gsm8k_train.jsonl
├── gsm8k_test.jsonl
├── math_train.jsonl
├── math_test.jsonl
├── mmlu_pro_train.jsonl
└── mmlu_pro_test.jsonl
```

For MMLU_Pro, sample the splits from HuggingFace `TIGER-Lab/MMLU-Pro`:

```bash
python scripts/sample_mmlu_pro.py            # default: 125 train / 500 test, seed 42
```

### Run Experiments

The full pipeline runs four phases automatically:

| Phase | Stage |
|------:|-------|
| 1     | Train the controller (Lagrangian-constrained REINFORCE) |
| 1.5   | Collect DAG traces with the converged controller |
| 2     | Train the pruning gate (offline MSE on Δ_k = R_k − R_N) |
| 3     | Test the controller (with the gate) on the held-out test set |

Run a single dataset (replace `MMLU_Pro` with `GSM8K` or `MATH` as needed):

```bash
python -m experiments.run_main --dataset MMLU_Pro
```

Run all three datasets concurrently in tmux windows:

```bash
./run_all_experiments.sh
```

### Results

Per-problem CSV results land in:

```
maas/ext/maas/scripts/optimized/experiments/<model_tag>/latency_weight_<w>/<DATASET>/test/round_1/*.csv
```

Each CSV row is one test problem and includes score, cost, latency and
critical-path token counts.
