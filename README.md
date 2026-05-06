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

### Run Experiments

The full pipeline runs four phases automatically:

| Phase | Stage |
|------:|-------|
| 1     | Train the controller (Lagrangian-constrained REINFORCE) |
| 1.5   | Collect DAG traces with the converged controller |
| 2     | Train the pruning gate (offline MSE on Δ_k = R_k − R_N) |
| 3     | Test the controller (with the gate) on the held-out test set |

Run a single dataset:

```bash
# GSM8K
python -m experiments.run_main --dataset GSM8K --acc_baseline 0.85

# MATH
python -m experiments.run_main --dataset MATH --acc_baseline 0.50

# MMLU_Pro
python -m experiments.run_main --dataset MMLU_Pro --acc_baseline 0.65
```

Run all three datasets concurrently in tmux windows:

```bash
./run_all_experiments.sh
```

### Useful Flags

| Flag | Effect |
|------|--------|
| `--test_only` | Skip Phase 1; load an existing controller checkpoint |
| `--train_only` | Skip Phase 3 |
| `--num_test_runs N` | Average over N independent test runs (default: 1) |
| `--checkpoint_path PATH` | Load a specific controller checkpoint |
| `--pruning_gate_path PATH` | Load a pre-trained gate (skips Phase 1.5 + 2) |
| `--latency_weight 0.005` | Latency penalty weight λ_t in the reward |
| `--cost_weight 3.0` | Cost penalty weight λ_c in the reward |
| `--acc_baseline 0.85` | Accuracy floor for the Lagrangian constraint |
| `--num_layers 4` | Number of controller layers |

### Results

Per-problem CSV results land in:

```
maas/ext/maas/scripts/optimized/experiments/<model_tag>/latency_weight_<w>/<DATASET>/test/round_1/*.csv
```

Each CSV row is one test problem and includes score, cost, latency and
critical-path token counts.

## Acknowledgements

Special thanks to [MaAS](https://github.com/bingreeky/MaAS) for the initial
codebase and prompt templates this project is built on.
