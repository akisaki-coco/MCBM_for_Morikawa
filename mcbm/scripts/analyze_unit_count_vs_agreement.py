import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BINS = [(1, 1), (2, 2), (3, 4), (5, 9), (10, 19), (20, None)]

ROW_FIELDNAMES = [
    "run",
    "split",
    "concept_id",
    "concept_name",
    "n_units",
    "status",
    "llm_sae_agreement",
    "n_samples",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the relationship between the number of merged SAE units per concept "
            "and llm_sae_agreement, including concepts skipped during annotation."
        )
    )
    parser.add_argument(
        "--annotation-dir",
        nargs="+",
        help="One or more annotation run directories (each containing annotation_summary_*.csv and run_info.json).",
    )
    parser.add_argument(
        "--concept-dir",
        nargs="+",
        help=(
            "Merged concept directories, one per --annotation-dir in the same order. "
            "If omitted, run_info.json['concept_dir'] of each annotation run is used."
        ),
    )
    parser.add_argument(
        "--sweep-summary",
        nargs="+",
        help="One or more sweep_summary.csv paths produced by sweep_expansion_factor.py or sweep_merge_threshold.py.",
    )
    parser.add_argument("--output-dir", help="Output directory (default: outputs/analysis/unit_count_vs_agreement).")
    return parser.parse_args()


def read_merged_concepts(concept_dir):
    path = Path(concept_dir) / "concept_names_merged.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    rows = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            units = [u for u in row["merged_from"].split(",") if u.strip()]
            rows[int(row["unit"])] = {"concept_name": row["concept_name"].strip(), "n_units": len(units)}
    return rows


def read_summary(path):
    if not path.exists():
        return None
    rows = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows[int(row["concept_id"])] = {
                "llm_sae_agreement": float(row["llm_sae_agreement"]),
                "n_samples": int(row["n_samples"]),
            }
    return rows


def _read_json_if_exists(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def resolve_concept_dir(annotation_dir, override):
    if override is not None:
        return Path(override)
    run_info_path = Path(annotation_dir) / "run_info.json"
    if not run_info_path.exists():
        raise FileNotFoundError(f"{run_info_path} not found; pass --concept-dir explicitly.")
    concept_dir = json.loads(run_info_path.read_text()).get("concept_dir")
    if not concept_dir:
        raise KeyError(f"{run_info_path} has no concept_dir; pass --concept-dir explicitly.")
    return Path(concept_dir)


def _splits_from_setting(split_setting):
    if split_setting == "all":
        return ["train", "test"]
    if split_setting in ("train", "test"):
        return [split_setting]
    return None


def resolve_run_entries(args):
    """Build (annotation_dir, concept_dir_override, target_splits) entries from CLI inputs."""
    if not args.annotation_dir and not args.sweep_summary:
        raise ValueError("Specify at least one of --annotation-dir or --sweep-summary.")
    if args.concept_dir is not None:
        if not args.annotation_dir or len(args.concept_dir) != len(args.annotation_dir):
            raise ValueError("--concept-dir must be given the same number of times as --annotation-dir.")

    entries = []
    if args.annotation_dir:
        for idx, ann_dir in enumerate(args.annotation_dir):
            ann_path = Path(ann_dir)
            ann_run_info = _read_json_if_exists(ann_path / "run_info.json")
            target_splits = _splits_from_setting(ann_run_info.get("split"))
            if target_splits is None:
                existing = [s for s in ("train", "test") if (ann_path / f"annotation_summary_{s}.csv").exists()]
                target_splits = existing if existing else ["train", "test"]
            entries.append((ann_path, args.concept_dir[idx] if args.concept_dir else None, target_splits))

    if args.sweep_summary:
        for summary_path_str in args.sweep_summary:
            summary_path = Path(summary_path_str)
            if not summary_path.exists():
                raise FileNotFoundError(f"Sweep summary not found: {summary_path}")
            with summary_path.open(newline="") as f:
                summary_rows = list(csv.DictReader(f))

            sweep_run_info = _read_json_if_exists(summary_path.parent / "run_info.json")
            sweep_splits = _splits_from_setting(sweep_run_info.get("split"))
            if sweep_splits is None:
                inferred = [
                    s
                    for s in ("train", "test")
                    if any(
                        (row.get("annotation_dir") or "").strip()
                        and (Path(row["annotation_dir"].strip()) / f"annotation_summary_{s}.csv").exists()
                        for row in summary_rows
                    )
                ]
                sweep_splits = inferred if inferred else ["train", "test"]

            for row in summary_rows:
                ann_str = (row.get("annotation_dir") or "").strip()
                if not ann_str:
                    continue
                concept_str = (row.get("merged_dir") or row.get("concept_dir") or "").strip() or None
                entries.append((Path(ann_str), concept_str, sweep_splits))

    if not entries:
        raise ValueError("No annotation runs found in the provided inputs.")
    return entries


def collect_rows(entries):
    rows = []
    for annotation_dir, concept_override, target_splits in entries:
        concept_dir = resolve_concept_dir(annotation_dir, concept_override)
        merged = read_merged_concepts(concept_dir)
        run_name = annotation_dir.name
        for split in target_splits:
            summary = read_summary(annotation_dir / f"annotation_summary_{split}.csv") or {}
            for concept_id, info in sorted(merged.items()):
                annotated = concept_id in summary
                rows.append(
                    {
                        "run": run_name,
                        "split": split,
                        "concept_id": concept_id,
                        "concept_name": info["concept_name"],
                        "n_units": info["n_units"],
                        "status": "annotated" if annotated else "skipped",
                        "llm_sae_agreement": summary[concept_id]["llm_sae_agreement"] if annotated else "",
                        "n_samples": summary[concept_id]["n_samples"] if annotated else "",
                    }
                )
    if not rows:
        raise ValueError("No concepts found across the specified runs.")
    return rows


def bin_label(low, high):
    if high is None:
        return f"{low}+"
    return str(low) if low == high else f"{low}-{high}"


def in_bin(n_units, low, high):
    return n_units >= low and (high is None or n_units <= high)


def correlation_text(units, agreements):
    if len(units) < 3 or len(set(units)) < 2 or len(set(agreements)) < 2:
        return "n/a (insufficient variation)"
    rho, rho_p = stats.spearmanr(units, agreements)
    r, r_p = stats.pearsonr(units, agreements)
    return f"Spearman rho={rho:+.3f} (p={rho_p:.3g}), Pearson r={r:+.3f} (p={r_p:.3g}), N={len(units)}"


def print_report(rows):
    for split in ("train", "test"):
        split_rows = [r for r in rows if r["split"] == split]
        if not split_rows:
            continue
        annotated = [r for r in split_rows if r["status"] == "annotated"]
        print(f"\n===== split={split} =====")
        print(
            "Correlation (n_units vs agreement, annotated only): "
            + correlation_text([r["n_units"] for r in annotated], [r["llm_sae_agreement"] for r in annotated])
        )
        multi = [r for r in annotated if r["n_units"] >= 2]
        print(
            "Correlation (n_units>=2 only): "
            + correlation_text([r["n_units"] for r in multi], [r["llm_sae_agreement"] for r in multi])
        )

        header = (
            f"{'n_units':>8} | {'concepts':>8} | {'skipped':>7} | {'skip_rate':>9} | "
            f"{'mean_agr':>8} | {'median':>6} | {'%>=0.55':>7} | {'%~0.50':>6}"
        )
        print(header)
        print("-" * len(header))
        for low, high in BINS:
            in_rows = [r for r in split_rows if in_bin(r["n_units"], low, high)]
            if not in_rows:
                continue
            ann = [r["llm_sae_agreement"] for r in in_rows if r["status"] == "annotated"]
            skipped = len(in_rows) - len(ann)
            mean_s = f"{np.mean(ann):.4f}" if ann else "-"
            med_s = f"{np.median(ann):.4f}" if ann else "-"
            ge55 = f"{np.mean([a >= 0.55 for a in ann]) * 100:.0f}%" if ann else "-"
            near50 = f"{np.mean([0.48 <= a <= 0.52 for a in ann]) * 100:.0f}%" if ann else "-"
            print(
                f"{bin_label(low, high):>8} | {len(in_rows):>8} | {skipped:>7} | "
                f"{skipped / len(in_rows) * 100:>8.0f}% | {mean_s:>8} | {med_s:>6} | {ge55:>7} | {near50:>6}"
            )

    runs = sorted({r["run"] for r in rows})
    if len(runs) > 1:
        print("\n===== per-run correlation (annotated only) =====")
        for run in runs:
            for split in ("train", "test"):
                sub = [r for r in rows if r["run"] == run and r["split"] == split and r["status"] == "annotated"]
                print(
                    f"{run} [{split}]: "
                    + correlation_text([r["n_units"] for r in sub], [r["llm_sae_agreement"] for r in sub])
                )


def save_outputs(rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "unit_count_vs_agreement.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for axis, split in zip(axes, ("train", "test")):
        ann = [r for r in rows if r["split"] == split and r["status"] == "annotated"]
        skipped = [r for r in rows if r["split"] == split and r["status"] == "skipped"]
        if ann:
            axis.scatter(
                [r["n_units"] for r in ann],
                [r["llm_sae_agreement"] for r in ann],
                alpha=0.6,
                label="annotated",
            )
        for idx, r in enumerate(skipped):
            axis.axvline(
                r["n_units"],
                color="red",
                alpha=0.2,
                linewidth=1,
                label="skipped" if idx == 0 else None,
            )
        axis.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (0.5)")
        axis.set_xscale("log")
        axis.set_xlabel("Number of merged SAE units (log scale)")
        axis.set_title(f"{split}")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")
    axes[0].set_ylabel("llm_sae_agreement")
    plt.tight_layout()
    plt.savefig(output_dir / "unit_count_vs_agreement.png", dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    entries = resolve_run_entries(args)
    rows = collect_rows(entries)
    print_report(rows)
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / "analysis" / "unit_count_vs_agreement"
    save_outputs(rows, output_dir)
    print(f"\nSaved per-concept table and scatter plot to {output_dir}")


if __name__ == "__main__":
    main()
