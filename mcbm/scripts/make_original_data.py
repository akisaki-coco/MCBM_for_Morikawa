import argparse
import json
from pathlib import Path

import pandas as pd

# CSVの画像とクラス，ラベルのカラム名
image_col = "filepath"
class_col = "cancer"
label_col = "split"


def parse_args():
    parser = argparse.ArgumentParser(description="Create original data CSVs and configs.")
    parser.add_argument("--csv", required=True, help="CSV with columns image_name,label (label: int from 0).")
    parser.add_argument("--dataset", default="palate", help="Base dataset name.")
    parser.add_argument("--config-template", required=True, help="Base config JSON (dataset/backbone are rewritten).")
    parser.add_argument("--output-dir", default="original_data", help="Output directory for original data CSVs and configs.")
    parser.add_argument("--seed", type=int, default = 42, help = "Random seed for shuffling the data.")
    return parser.parse_args()

def main():
    args = parse_args()
    data = pd.read_csv(args.csv)

    if not {image_col, class_col, label_col}.issubset(data.columns):
        raise ValueError(f"CSV must have columns {image_col},{class_col},{label_col}.")