import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold


def parse_args():
    parser = argparse.ArgumentParser(description="Create stratified k-fold CSVs and per-fold configs.")
    parser.add_argument("--csv", required=True, help="CSV with columns image_name,label (label: int from 0).")
    parser.add_argument("--dataset", default="mydata", help="Base dataset name.")
    parser.add_argument("--config-template", required=True, help="Base config JSON (dataset/backbone are rewritten).")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold-dir", help="Output directory for fold CSVs. Default: <csv dir>/folds")
    parser.add_argument("--config-dir", default="configs")
    return parser.parse_args()


def main():
    args = parse_args()
    data = pd.read_csv(args.csv)
    if not {"image_name", "label"}.issubset(data.columns):
        raise ValueError("CSV must have columns image_name,label.")

    fold_dir = Path(args.fold_dir) if args.fold_dir else Path(args.csv).parent / "folds"
    fold_dir.mkdir(parents=True, exist_ok=True)
    config_dir = Path(args.config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    template = json.loads(Path(args.config_template).read_text())

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    for fold, (train_idx, test_idx) in enumerate(skf.split(data, data["label"])):
        name = f"{args.dataset}_fold{fold}"
        data.iloc[train_idx].to_csv(fold_dir / f"{name}_train.csv", index=False)
        data.iloc[test_idx].to_csv(fold_dir / f"{name}_test.csv", index=False)

        config = dict(template)
        config["dataset"] = name
        config["backbone"] = f"{template['backbone'].split('_fold')[0]}_fold{fold}"
        (config_dir / f"{name}.json").write_text(json.dumps(config, indent=4))
        print(f"{name}: train={len(train_idx)}, test={len(test_idx)}")


if __name__ == "__main__":
    main()
