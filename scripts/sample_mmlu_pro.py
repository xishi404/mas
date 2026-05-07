#!/usr/bin/env python
"""Sample MMLU-Pro train/test splits with a seeded random shuffle.

Source: HuggingFace `TIGER-Lab/MMLU-Pro` test split (12,032 questions).
Output:
  maas/ext/maas/data/mmlu_pro_train.jsonl   (default 125 questions)
  maas/ext/maas/data/mmlu_pro_test.jsonl    (default 500 questions, disjoint from train)

The full pool is shuffled once with the seed; the first `train_size` rows go to train,
the next `test_size` rows go to test. Per-category counts will roughly mirror the
source distribution.
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

OUTPUT_FIELDS = ["question_id", "question", "options", "answer", "answer_index",
                 "category", "src"]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train_size", type=int, default=125)
    parser.add_argument("--test_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="maas/ext/maas/data")
    args = parser.parse_args()

    print("Downloading TIGER-Lab/MMLU-Pro test split...")
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    print(f"Loaded {len(ds)} questions")

    pool = [{k: row[k] for k in OUTPUT_FIELDS} for row in ds]
    if len(pool) < args.train_size + args.test_size:
        raise RuntimeError(f"Pool has only {len(pool)} entries, "
                           f"need {args.train_size + args.test_size}")

    rng = random.Random(args.seed)
    rng.shuffle(pool)
    train = pool[:args.train_size]
    test = pool[args.train_size:args.train_size + args.test_size]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "mmlu_pro_train.jsonl"
    test_path = out_dir / "mmlu_pro_test.jsonl"

    with train_path.open("w") as f:
        for row in train:
            f.write(json.dumps(row) + "\n")
    with test_path.open("w") as f:
        for row in test:
            f.write(json.dumps(row) + "\n")

    from collections import Counter
    print(f"\nWrote {len(train)} train -> {train_path}")
    print(f"Wrote {len(test)} test  -> {test_path}")
    print(f"\nTrain category distribution:")
    for c, n in sorted(Counter(r['category'] for r in train).items()):
        print(f"  {c}: {n}")
    print(f"\nTest category distribution:")
    for c, n in sorted(Counter(r['category'] for r in test).items()):
        print(f"  {c}: {n}")


if __name__ == "__main__":
    main()
