# LaMaS

Critical-path-aware Lagrangian optimization of LLM-based multi-agent workflows.

## Reproducing Experiments

### Prerequisites

```bash
pip install -r requirements.txt
```

Configure API keys in `config/config2.yaml` (copy from `config/config2.example.yaml`).

### Dataset Setup

Place the GSM8K, MATH and MMLU_Pro datasets under `lamas/ext/lamas/data/`:

```
lamas/ext/lamas/data/
├── gsm8k_train.jsonl
├── gsm8k_test.jsonl
├── math_train.jsonl
├── math_test.jsonl
├── mmlu_pro_train.jsonl
└── mmlu_pro_test.jsonl
```

### Run Experiments

```bash
python -m experiments.run_main --dataset MMLU_Pro
```

### Results

Per-problem CSV results land in:

```
lamas/ext/lamas/scripts/optimized/experiments/<model_tag>/latency_weight_<w>/<DATASET>/test/round_1/*.csv
```

Each CSV row is one test problem and includes score, cost, latency and
critical-path token counts.
