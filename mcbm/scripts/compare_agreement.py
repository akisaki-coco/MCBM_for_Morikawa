import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SUMMARY_STATS_FIELDNAMES = [
    "condition",
    "n",
    "mean",
    "std",
    "median",
    "q25",
    "q75",
    "min",
    "max",
    "mean_pct",
    "std_pct",
]

PAIRED_FIELDNAMES = [
    "concept_id",
    "default_concept_name",
    "default_agreement",
    "random_concept_name",
    "random_agreement",
    "diff",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate, statistically compare, and plot SAE-MLLM agreement across conditions."
    )
    parser.add_argument(
        "--default",
        dest="default_input",
        required=True,
        help="Path to default annotation_summary_<split>.csv or annotation run directory.",
    )
    parser.add_argument(
        "--random",
        dest="random_input",
        help="Optional path to random-word control annotation_summary_<split>.csv or annotation run directory.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="test",
        help="Split used when an input path is a directory (default: test).",
    )
    parser.add_argument(
        "--default-label",
        default="Default\n(Named)",
        help="Plot label for the default condition.",
    )
    parser.add_argument(
        "--random-label",
        default="Control\n(Random Words)",
        help="Plot label for the random-word control condition.",
    )
    parser.add_argument(
        "--all-concepts",
        action="store_true",
        help="When comparing two runs, evaluate all concepts in each run instead of aligning on common concept_ids.",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory (default: outputs/analysis/agreement_comparison/<timestamp>).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output PNG DPI (default: 200).",
    )
    return parser.parse_args()


def resolve_summary_csv(path_str, split):
    path = Path(path_str)
    if path.is_dir():
        csv_path = path / f"annotation_summary_{split}.csv"
    else:
        csv_path = path
    if not csv_path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {csv_path}")
    return csv_path


def load_summary_csv(csv_path):
    rows = {}
    with Path(csv_path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"concept_id", "concept_name", "llm_sae_agreement", "n_samples"}
        fieldnames = set(reader.fieldnames or [])
        if not required.issubset(fieldnames):
            raise ValueError(f"{csv_path} must contain columns: {sorted(required)}.")
        for row in reader:
            cid = int(row["concept_id"])
            rows[cid] = {
                "concept_id": cid,
                "concept_name": row["concept_name"].strip(),
                "llm_sae_agreement": float(row["llm_sae_agreement"]),
                "n_samples": int(row["n_samples"]),
            }
    if not rows:
        raise ValueError(f"{csv_path} contains no rows.")
    return rows


def compute_stats(values):
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        raise ValueError("Cannot compute statistics on empty values.")
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n >= 2 else None
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "median": float(np.median(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean_pct": mean * 100.0,
        "std_pct": std * 100.0 if std is not None else None,
    }


def format_stats_line(label, st):
    clean_label = label.replace("\n", " ")
    if st["std"] is not None:
        return (
            f"{clean_label}: "
            f"{st['mean_pct']:.1f}% ± {st['std_pct']:.1f}% "
            f"({st['mean']:.4f} ± {st['std']:.4f}), "
            f"median={st['median'] * 100:.1f}%, "
            f"IQR=[{st['q25'] * 100:.1f}%, {st['q75'] * 100:.1f}%], "
            f"range=[{st['min'] * 100:.1f}%, {st['max'] * 100:.1f}%], "
            f"n={st['n']}"
        )
    return (
        f"{clean_label}: "
        f"{st['mean_pct']:.1f}% ({st['mean']:.4f}), "
        f"std=n/a (n={st['n']})"
    )


def compute_paired_tests(default_vals, random_vals):
    default_arr = np.asarray(default_vals, dtype=np.float64)
    random_arr = np.asarray(random_vals, dtype=np.float64)
    if default_arr.size < 2:
        return {"mode": "paired", "note": "n/a (n < 2)"}
    diff = default_arr - random_arr
    if np.all(diff == 0):
        return {"mode": "paired", "note": "n/a (zero differences)"}

    result = {"mode": "paired", "mean_diff": float(np.mean(diff))}
    if np.std(diff, ddof=1) == 0:
        result["ttest_rel"] = "n/a (zero variance in differences)"
    else:
        t_stat, t_p = stats.ttest_rel(default_arr, random_arr)
        result["ttest_rel"] = {"statistic": float(t_stat), "p_value": float(t_p)}

    w_stat, w_p = stats.wilcoxon(diff)
    result["wilcoxon"] = {"statistic": float(w_stat), "p_value": float(w_p)}
    return result


def compute_unpaired_tests(default_vals, random_vals):
    default_arr = np.asarray(default_vals, dtype=np.float64)
    random_arr = np.asarray(random_vals, dtype=np.float64)
    if default_arr.size < 2 or random_arr.size < 2:
        return {"mode": "unpaired", "note": "n/a (each group requires n >= 2)"}

    result = {"mode": "unpaired", "mean_diff": float(np.mean(default_arr) - np.mean(random_arr))}
    if np.std(default_arr, ddof=1) == 0 and np.std(random_arr, ddof=1) == 0:
        result["welch_t"] = "n/a (zero variance in both groups)"
    else:
        t_stat, t_p = stats.ttest_ind(default_arr, random_arr, equal_var=False)
        result["welch_t"] = {"statistic": float(t_stat), "p_value": float(t_p)}

    u_stat, u_p = stats.mannwhitneyu(default_arr, random_arr, alternative="two-sided")
    result["mannwhitneyu"] = {"statistic": float(u_stat), "p_value": float(u_p)}
    return result


def format_test_result(test_info):
    if "note" in test_info:
        return [f"Statistical test ({test_info['mode']}): {test_info['note']}"]
    lines = [f"Mean difference (Default - Random): {test_info['mean_diff'] * 100:+.2f} pp"]
    if test_info["mode"] == "paired":
        t_res = test_info["ttest_rel"]
        w_res = test_info["wilcoxon"]
        t_str = (
            f"t={t_res['statistic']:+.3f}, p={t_res['p_value']:.4g}"
            if isinstance(t_res, dict)
            else t_res
        )
        lines.append(f"- Paired t-test: {t_str}")
        lines.append(f"- Wilcoxon signed-rank: W={w_res['statistic']:.3f}, p={w_res['p_value']:.4g}")
    else:
        t_res = test_info["welch_t"]
        u_res = test_info["mannwhitneyu"]
        t_str = (
            f"t={t_res['statistic']:+.3f}, p={t_res['p_value']:.4g}"
            if isinstance(t_res, dict)
            else t_res
        )
        lines.append(f"- Welch's t-test: {t_str}")
        lines.append(f"- Mann-Whitney U: U={u_res['statistic']:.3f}, p={u_res['p_value']:.4g}")
    return lines


def save_summary_stats_csv(path, stats_rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_STATS_FIELDNAMES)
        writer.writeheader()
        for label, st in stats_rows:
            writer.writerow(
                {
                    "condition": label.replace("\n", " "),
                    "n": st["n"],
                    "mean": f"{st['mean']:.6f}",
                    "std": f"{st['std']:.6f}" if st["std"] is not None else "",
                    "median": f"{st['median']:.6f}",
                    "q25": f"{st['q25']:.6f}",
                    "q75": f"{st['q75']:.6f}",
                    "min": f"{st['min']:.6f}",
                    "max": f"{st['max']:.6f}",
                    "mean_pct": f"{st['mean_pct']:.3f}",
                    "std_pct": f"{st['std_pct']:.3f}" if st["std_pct"] is not None else "",
                }
            )


def save_paired_csv(path, common_ids, default_map, random_map):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PAIRED_FIELDNAMES)
        writer.writeheader()
        for cid in common_ids:
            d_row = default_map[cid]
            r_row = random_map[cid]
            diff = d_row["llm_sae_agreement"] - r_row["llm_sae_agreement"]
            writer.writerow(
                {
                    "concept_id": cid,
                    "default_concept_name": d_row["concept_name"],
                    "default_agreement": f"{d_row['llm_sae_agreement']:.6f}",
                    "random_concept_name": r_row["concept_name"],
                    "random_agreement": f"{r_row['llm_sae_agreement']:.6f}",
                    "diff": f"{diff:.6f}",
                }
            )


def plot_agreement_comparison(path, groups, dpi):
    """Plot boxplot + jittered points + mean diamond + 50% chance baseline."""
    fig_width = 4.2 if len(groups) == 1 else 5.8
    fig, ax = plt.subplots(figsize=(fig_width, 4.6))

    positions = list(range(1, len(groups) + 1))
    values_pct = [np.asarray(g["values"], dtype=np.float64) * 100.0 for g in groups]

    bp = ax.boxplot(
        values_pct,
        positions=positions,
        widths=0.42,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "#1f2937", "linewidth": 1.8},
        whiskerprops={"color": "#4b5563", "linewidth": 1.2},
        capprops={"color": "#4b5563", "linewidth": 1.2},
    )
    box_colors = ["#dbeafe", "#e5e7eb"]
    edge_colors = ["#2563eb", "#4b5563"]
    for idx, box in enumerate(bp["boxes"]):
        box.set_facecolor(box_colors[idx % len(box_colors)])
        box.set_edgecolor(edge_colors[idx % len(edge_colors)])
        box.set_linewidth(1.4)

    rng = np.random.default_rng(0)
    for idx, (pos, vals) in enumerate(zip(positions, values_pct)):
        jitter = rng.normal(0.0, 0.045, size=len(vals)) if len(vals) > 1 else np.array([0.0])
        ax.scatter(
            pos + jitter,
            vals,
            color=edge_colors[idx % len(edge_colors)],
            alpha=0.6,
            s=24,
            zorder=3,
        )

    means = [float(np.mean(v)) for v in values_pct]
    ax.scatter(
        positions,
        means,
        marker="D",
        color="#f59e0b",
        edgecolor="#78350f",
        s=55,
        zorder=4,
        label="Mean",
    )

    ax.axhline(
        50.0,
        color="#dc2626",
        linestyle="--",
        linewidth=1.4,
        zorder=2,
        label="Chance rate (50%)",
    )

    for pos, group, vals in zip(positions, groups, values_pct):
        st = group["stats"]
        if st["std_pct"] is not None:
            badge = f"{st['mean_pct']:.1f}% ± {st['std_pct']:.1f}%"
        else:
            badge = f"{st['mean_pct']:.1f}%"
        y_text = min(float(np.max(vals)) + 4.5, 102.0)
        ax.text(
            pos,
            y_text,
            badge,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#111827",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
            zorder=5,
        )

    xticklabels = [f"{g['label']}\n(n={g['stats']['n']})" for g in groups]
    ax.set_xticks(positions)
    ax.set_xticklabels(xticklabels, fontsize=10)
    ax.set_ylabel("SAE-MLLM Agreement (%)", fontsize=11)
    ax.set_ylim(0, 109)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        fontsize=9.5,
    )

    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close(fig)


def main():
    args = parse_args()
    default_csv = resolve_summary_csv(args.default_input, args.split)
    default_map = load_summary_csv(default_csv)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("outputs") / "analysis" / "agreement_comparison" / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = []
    common_ids = []
    random_map = None
    random_csv = None
    test_info = None

    if args.random_input is None:
        default_vals = [default_map[cid]["llm_sae_agreement"] for cid in sorted(default_map)]
        default_stats = compute_stats(default_vals)
        groups.append({"label": args.default_label, "values": default_vals, "stats": default_stats})
    else:
        random_csv = resolve_summary_csv(args.random_input, args.split)
        random_map = load_summary_csv(random_csv)
        common_ids = sorted(set(default_map) & set(random_map))

        if not args.all_concepts:
            if not common_ids:
                raise ValueError(
                    "No common concept_ids found between --default and --random. "
                    "Pass --all-concepts to compare all concepts without pairing."
                )
            dropped_default = len(default_map) - len(common_ids)
            dropped_random = len(random_map) - len(common_ids)
            if dropped_default > 0 or dropped_random > 0:
                print(
                    f"Aligned on {len(common_ids)} common concept_ids "
                    f"(excluded {dropped_default} from default, {dropped_random} from random)."
                )
            default_vals = [default_map[cid]["llm_sae_agreement"] for cid in common_ids]
            random_vals = [random_map[cid]["llm_sae_agreement"] for cid in common_ids]
            test_info = compute_paired_tests(default_vals, random_vals)
        else:
            default_vals = [default_map[cid]["llm_sae_agreement"] for cid in sorted(default_map)]
            random_vals = [random_map[cid]["llm_sae_agreement"] for cid in sorted(random_map)]
            test_info = compute_unpaired_tests(default_vals, random_vals)

        groups.append(
            {
                "label": args.default_label,
                "values": default_vals,
                "stats": compute_stats(default_vals),
            }
        )
        groups.append(
            {
                "label": args.random_label,
                "values": random_vals,
                "stats": compute_stats(random_vals),
            }
        )

    print("=== SAE-MLLM Agreement Summary ===")
    for group in groups:
        print(format_stats_line(group["label"], group["stats"]))
    if test_info is not None:
        print()
        for line in format_test_result(test_info):
            print(line)

    stats_csv_path = output_dir / "summary_stats.csv"
    save_summary_stats_csv(stats_csv_path, [(g["label"], g["stats"]) for g in groups])

    paired_csv_path = None
    if random_map is not None and common_ids:
        paired_csv_path = output_dir / "paired_concepts.csv"
        save_paired_csv(paired_csv_path, common_ids, default_map, random_map)

    png_path = output_dir / "agreement_comparison.png"
    plot_agreement_comparison(png_path, groups, dpi=args.dpi)

    run_info_path = output_dir / "run_info.json"
    run_info = {
        "timestamp": timestamp,
        "split": args.split,
        "default_csv": str(default_csv),
        "random_csv": str(random_csv) if random_csv is not None else None,
        "paired_alignment": (not args.all_concepts) if random_map is not None else None,
        "n_common_concepts": len(common_ids) if random_map is not None else None,
        "statistical_test": test_info,
    }
    run_info_path.write_text(json.dumps(run_info, indent=2), encoding="utf-8")

    print(f"\nSaved outputs to {output_dir}:")
    print(f"- {stats_csv_path.name}")
    if paired_csv_path is not None:
        print(f"- {paired_csv_path.name}")
    print(f"- {png_path.name}")
    print(f"- {run_info_path.name}")


if __name__ == "__main__":
    main()
