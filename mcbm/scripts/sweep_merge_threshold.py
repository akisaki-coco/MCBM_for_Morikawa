import argparse
import copy
import csv
import gc
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from mcbm.data.utils import get_dataset_domain
from mcbm.scripts.annotate_concepts import annotate_concepts, get_annotation_budget
from mcbm.scripts.merge_concepts import (
    apply_context,
    build_merged_output,
    get_embeddings,
    load_run_info,
    merge_identical_names,
    read_concepts,
    resolve_api_key as resolve_embedding_api_key,
    save_merged_output,
)
from mcbm.scripts.name_concepts import (
    resolve_api_key as resolve_llm_api_key,
    validate_llm_options,
)
from mcbm.scripts.sweep_expansion_factor import read_annotation_stats, sanitize_model_name


DEFAULT_THRESHOLDS = "0.70,0.75,0.80,0.85,0.88,0.90,0.92,0.95,0.98"
DEFAULT_EMBEDDING_MODEL = "qwen3-embedding:0.6b"
DEFAULT_LLM_MODEL = "qwen3.5:27b"

SUMMARY_FIELDNAMES = [
    "merge_similarity_threshold",
    "raw_concepts",
    "exact_unique_concepts",
    "merged_concepts",
    "multi_name_clusters",
    "max_units_per_concept",
    "config_path",
    "merged_dir",
    "annotation_dir",
    "train_annotated_concepts",
    "train_mean_llm_sae_agreement",
    "test_annotated_concepts",
    "test_mean_llm_sae_agreement",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep merge_similarity_threshold using cached embeddings from a concept naming run."
    )
    parser.add_argument("--config", required=True, help="Path to base dataset config JSON.")
    parser.add_argument(
        "--concept-dir",
        help=(
            "Concept naming run directory containing concept_names.csv and run_info.json. "
            "If omitted, the latest valid run under outputs/concepts/<dataset>/ is used."
        ),
    )
    parser.add_argument(
        "--thresholds",
        default=DEFAULT_THRESHOLDS,
        help=f"Comma-separated merge_similarity_threshold values in (0, 1] (default: {DEFAULT_THRESHOLDS}).",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for sweep outputs. Defaults to outputs/sweeps/<dataset>/sweep_merge_<timestamp>.",
    )
    parser.add_argument(
        "--embedding-provider",
        choices=["openai", "gemini", "ollama"],
        default="ollama",
        help="Embedding provider for concept merging (default: ollama).",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=(
            f"Embedding model name for concept merging (default: {DEFAULT_EMBEDDING_MODEL}; "
            "specify e.g. text-embedding-3-large for openai or gemini-embedding-001 for gemini)."
        ),
    )
    parser.add_argument(
        "--run-annotate",
        action="store_true",
        help="Also run annotate_concepts for each threshold's merged concepts.",
    )
    parser.add_argument("--device", help="Torch device override when --run-annotate is set.")
    parser.add_argument(
        "--llm-provider",
        choices=["openai", "gemini", "ollama"],
        default="ollama",
        help="LLM provider when --run-annotate is set (default: ollama).",
    )
    parser.add_argument(
        "--llm-model",
        default=DEFAULT_LLM_MODEL,
        help=f"LLM model name when --run-annotate is set (default: {DEFAULT_LLM_MODEL}).",
    )
    parser.add_argument(
        "--split",
        choices=["all", "train", "test"],
        default="all",
        help="Annotation split when --run-annotate is set (default: all).",
    )
    parser.add_argument("--reasoning-effort", choices=["none", "minimal", "low", "medium", "high", "xhigh"])
    parser.add_argument("--thinking-level", choices=["minimal", "low", "medium", "high"])
    parser.add_argument("--save-images", action="store_true", help="Save annotation grids when --run-annotate is set.")
    return parser.parse_args()


def parse_thresholds(value):
    """Parse and validate a comma-separated list of similarity thresholds in (0, 1]."""
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("--thresholds must contain at least one threshold in (0, 1].")

    thresholds = []
    seen = set()
    for item in items:
        try:
            numeric = float(item)
        except ValueError as exc:
            raise ValueError(f"Invalid threshold '{item}': must be a number.") from exc
        if not math.isfinite(numeric) or numeric <= 0.0 or numeric > 1.0:
            raise ValueError(f"Threshold must be finite and in (0, 1], got {item}.")
        if numeric in seen:
            raise ValueError(f"Duplicate threshold detected: {numeric}")
        seen.add(numeric)
        thresholds.append(numeric)
    return thresholds


def format_threshold_tag(threshold):
    """Format a similarity threshold as a filesystem-safe tag."""
    return f"{threshold:g}".replace(".", "_").replace("-", "_")


def resolve_concept_dir(base_config, concept_dir_arg):
    """Validate --concept-dir or auto-select the latest valid run for the dataset."""
    if concept_dir_arg is not None:
        concept_dir = Path(concept_dir_arg)
        if not concept_dir.is_dir():
            raise NotADirectoryError(f"Concept directory not found: {concept_dir}")
        for required_name in ("concept_names.csv", "run_info.json"):
            if not (concept_dir / required_name).exists():
                raise FileNotFoundError(f"Required file {required_name} not found in {concept_dir}")
        return concept_dir

    dataset = base_config["dataset"]
    parent = Path("outputs") / "concepts" / dataset
    if not parent.is_dir():
        raise FileNotFoundError(
            f"Concept directory parent not found: {parent}. Specify --concept-dir explicitly."
        )

    candidates = [
        candidate
        for candidate in parent.iterdir()
        if candidate.is_dir()
        and (candidate / "concept_names.csv").exists()
        and (candidate / "run_info.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No valid concept run directory (with concept_names.csv and run_info.json) found under {parent}. "
            "Specify --concept-dir explicitly."
        )

    selected = max(candidates, key=lambda p: (p.stat().st_mtime_ns, p.name))
    print(f"Auto-selected concept_dir: {selected}")
    return selected


def validate_sweep_options(args, base_config, base_run_info):
    """Validate provider models, API keys, and annotation prerequisites upfront."""
    if args.embedding_provider in ("openai", "gemini"):
        if args.embedding_model == DEFAULT_EMBEDDING_MODEL:
            raise ValueError(f"Specify --embedding-model when using --embedding-provider {args.embedding_provider}.")
        resolve_embedding_api_key(args.embedding_provider)

    if not args.run_annotate:
        return None, None

    validate_llm_options(args)
    if args.llm_provider in ("openai", "gemini"):
        if args.llm_model == DEFAULT_LLM_MODEL:
            raise ValueError(f"Specify --llm-model when using --llm-provider {args.llm_provider}.")
        resolve_llm_api_key(args.llm_provider)

    if args.split in ("all", "train"):
        get_annotation_budget(base_config, "train")
    if args.split in ("all", "test"):
        get_annotation_budget(base_config, "test")

    activation_dir = base_run_info.get("activation_dir")
    sae_dir = base_run_info.get("sae_dir")
    if not activation_dir or not Path(activation_dir).exists():
        raise FileNotFoundError(f"activation_dir from run_info.json not found: {activation_dir}")
    if not sae_dir or not Path(sae_dir).exists():
        raise FileNotFoundError(f"sae_dir from run_info.json not found: {sae_dir}")
    return Path(activation_dir), Path(sae_dir)


def prepare_sweep_dir(base_config, output_dir):
    if output_dir is not None:
        return Path(output_dir)
    dataset = base_config["dataset"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs") / "sweeps" / dataset / f"sweep_merge_{timestamp}"


def write_text_atomic(path, content):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content)
    tmp_path.replace(path)


def save_csv_atomic(path, fieldnames, rows):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp_path.replace(path)


def compute_and_save_similarity_pairs(sweep_dir, intermediate, sim_matrix):
    """Save pairwise cosine similarities sorted descending and print distribution stats."""
    pair_rows = []
    if sim_matrix is not None:
        for i in range(len(intermediate)):
            for j in range(i + 1, len(intermediate)):
                pair_rows.append(
                    {
                        "concept_a": intermediate[i]["concept_name"],
                        "concept_b": intermediate[j]["concept_name"],
                        "cosine_similarity": float(sim_matrix[i, j]),
                        "units_a": intermediate[i]["merged_from"],
                        "units_b": intermediate[j]["merged_from"],
                    }
                )
        pair_rows.sort(key=lambda row: row["cosine_similarity"], reverse=True)

    save_csv_atomic(
        sweep_dir / "similarity_pairs.csv",
        ["concept_a", "concept_b", "cosine_similarity", "units_a", "units_b"],
        pair_rows,
    )

    if not pair_rows:
        print("Only 1 unique concept after exact-name merging; no similarity pairs to analyze.")
        return

    sims = np.array([row["cosine_similarity"] for row in pair_rows], dtype=np.float64)
    pcts = np.percentile(sims, [0, 50, 75, 90, 95, 99, 100])
    print(
        "Pairwise cosine similarity distribution "
        f"(pairs={len(pair_rows)}): "
        f"min={pcts[0]:.4f}, p50={pcts[1]:.4f}, p75={pcts[2]:.4f}, "
        f"p90={pcts[3]:.4f}, p95={pcts[4]:.4f}, p99={pcts[5]:.4f}, max={pcts[6]:.4f}"
    )
    print("Top 20 most similar concept pairs:")
    for rank, row in enumerate(pair_rows[:20], start=1):
        print(f"  {rank:>2}. {row['cosine_similarity']:.4f} | '{row['concept_a']}' <-> '{row['concept_b']}'")


def cluster_concepts_for_threshold(intermediate, embeddings, threshold):
    """Run agglomerative clustering matching merge_concepts.py (or keep all when t=1.0)."""
    if len(intermediate) == 1 or threshold == 1.0:
        labels = np.arange(len(intermediate), dtype=int)
        return copy.deepcopy(intermediate), labels

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="complete",
        distance_threshold=1.0 - threshold,
    )
    labels = clustering.fit_predict(embeddings)
    merged = build_merged_output(intermediate, embeddings, labels)
    return merged, labels


def build_cluster_details(intermediate, merged, labels, sim_matrix):
    """Collect details of clusters that combined at least two exact-name concepts."""
    label_to_indices = {}
    for idx, label in enumerate(labels):
        label_to_indices.setdefault(int(label), []).append(idx)

    # build_merged_output orders merged rows by sorted(set(labels))
    sorted_cluster_ids = sorted(label_to_indices)
    details = []
    for merged_row, cluster_id in zip(merged, sorted_cluster_ids):
        indices = label_to_indices[cluster_id]
        if len(indices) < 2:
            continue

        pair_sims = [
            float(sim_matrix[a, b])
            for pos_a, a in enumerate(indices)
            for b in indices[pos_a + 1 :]
        ]
        details.append(
            {
                "unit": int(merged_row["unit"]),
                "representative": merged_row["concept_name"],
                "merged_from_units": merged_row["merged_from"],
                "member_count": len(indices),
                "members": [intermediate[idx]["concept_name"] for idx in indices],
                "min_similarity": min(pair_sims),
                "mean_similarity": sum(pair_sims) / len(pair_sims),
            }
        )
    return details


def format_cluster_details_text(all_threshold_details):
    lines = []
    for entry in all_threshold_details:
        lines.append(
            f"=== threshold={entry['threshold']:g} | "
            f"merged_concepts={entry['merged_concepts']} | "
            f"multi_name_clusters={len(entry['clusters'])} | "
            f"max_units={entry['max_units_per_concept']} ==="
        )
        if not entry["clusters"]:
            lines.append("  (no multi-name clusters merged)")
        for cluster in entry["clusters"]:
            members_str = " | ".join(cluster["members"])
            lines.append(
                f"  unit {cluster['unit']:>2}: '{cluster['representative']}' "
                f"(n={cluster['member_count']}, min_sim={cluster['min_similarity']:.4f}, "
                f"mean_sim={cluster['mean_similarity']:.4f}) <- [{members_str}]"
            )
        lines.append("")
    return "\n".join(lines)


def count_units_in_row(merged_row):
    return len([item for item in merged_row["merged_from"].split(",") if item.strip()])


def print_summary_table(rows):
    print("\n=== Merge Threshold Sweep Summary ===")
    header = (
        f"{'threshold':>9} | {'exact':>5} | {'merged':>6} | "
        f"{'clusters>=2':>11} | {'max_units':>9} | {'train_agree':>11} | {'test_agree':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        tr = (
            f"{row['train_mean_llm_sae_agreement']:.4f}"
            if row["train_mean_llm_sae_agreement"] is not None
            else "-"
        )
        te = (
            f"{row['test_mean_llm_sae_agreement']:.4f}"
            if row["test_mean_llm_sae_agreement"] is not None
            else "-"
        )
        print(
            f"{row['merge_similarity_threshold']:>9g} | "
            f"{row['exact_unique_concepts']:>5} | "
            f"{row['merged_concepts']:>6} | "
            f"{row['multi_name_clusters']:>11} | "
            f"{row['max_units_per_concept']:>9} | "
            f"{tr:>11} | "
            f"{te:>10}"
        )


def run_sweep(args):
    base_config = json.loads(Path(args.config).read_text())
    thresholds = parse_thresholds(args.thresholds)
    concept_dir = resolve_concept_dir(base_config, args.concept_dir)
    base_run_info = load_run_info(concept_dir)
    activation_dir, sae_dir = validate_sweep_options(args, base_config, base_run_info)

    device = torch.device(args.device or base_config.get("device", "cuda")) if args.run_annotate else None
    sweep_dir = prepare_sweep_dir(base_config, args.output_dir)
    configs_dir = sweep_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    sweep_run_info = {
        "base_config": str(args.config),
        "concept_dir": str(concept_dir),
        "thresholds": thresholds,
        "embedding_provider": args.embedding_provider,
        "embedding_model": args.embedding_model,
        "run_annotate": bool(args.run_annotate),
        "llm_provider": args.llm_provider if args.run_annotate else None,
        "llm_model": args.llm_model if args.run_annotate else None,
        "split": args.split if args.run_annotate else None,
        "started_at": datetime.now().isoformat(),
    }
    write_text_atomic(sweep_dir / "run_info.json", json.dumps(sweep_run_info, indent=2))

    raw_concepts = read_concepts(concept_dir / "concept_names.csv")
    intermediate = merge_identical_names(raw_concepts)
    domain = get_dataset_domain(base_config["dataset"])
    concepts_with_context = [apply_context(row["concept_name"], domain) for row in intermediate]

    print(f"Loaded {len(raw_concepts)} concept names from {concept_dir}.")
    print(f"After exact-name merging: {len(intermediate)} unique concept names.")
    print(f"Embedding provider: {args.embedding_provider} ({args.embedding_model})")

    embeddings = None
    sim_matrix = None
    if len(intermediate) >= 2:
        embeddings = get_embeddings(concepts_with_context, args.embedding_provider, args.embedding_model)
        sim_matrix = cosine_similarity(embeddings)

    compute_and_save_similarity_pairs(sweep_dir, intermediate, sim_matrix)

    summary_rows = []
    all_cluster_details = []
    dataset = base_config["dataset"]
    model_name = sanitize_model_name(args.llm_provider, args.llm_model)

    for index, threshold in enumerate(thresholds, start=1):
        th_tag = format_threshold_tag(threshold)
        threshold_config = copy.deepcopy(base_config)
        threshold_config["merge_similarity_threshold"] = threshold

        config_path = configs_dir / f"config_th_{th_tag}.json"
        write_text_atomic(config_path, json.dumps(threshold_config, indent=4))

        merged_dir = sweep_dir / f"merged_th_{th_tag}"
        merged_dir.mkdir(parents=True, exist_ok=True)

        merged, labels = cluster_concepts_for_threshold(intermediate, embeddings, threshold)
        threshold_run_info = copy.deepcopy(base_run_info)
        threshold_run_info["embedding_provider"] = args.embedding_provider
        threshold_run_info["embedding_model"] = args.embedding_model
        threshold_run_info["merge_similarity_threshold"] = threshold
        save_merged_output(merged_dir, concept_dir, threshold_run_info, merged)

        clusters = build_cluster_details(intermediate, merged, labels, sim_matrix)
        max_units = max(count_units_in_row(row) for row in merged)
        write_text_atomic(merged_dir / "cluster_details.json", json.dumps(clusters, indent=2))

        all_cluster_details.append(
            {
                "threshold": threshold,
                "merged_concepts": len(merged),
                "max_units_per_concept": max_units,
                "clusters": clusters,
            }
        )
        write_text_atomic(sweep_dir / "cluster_details.json", json.dumps(all_cluster_details, indent=2))
        write_text_atomic(sweep_dir / "cluster_details.txt", format_cluster_details_text(all_cluster_details))

        annotation_dir = None
        train_count = 0
        train_agreement = None
        test_count = 0
        test_agreement = None

        if args.run_annotate:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            annotation_dir = (
                Path("outputs")
                / "annotations"
                / dataset
                / f"annotations_{dataset}_{model_name}_th{th_tag}_{timestamp}"
            )
            print(f"\n[{index}/{len(thresholds)}] Annotating threshold={threshold} -> {annotation_dir}")
            annotator_args = argparse.Namespace(
                config=str(config_path),
                concept_dir=str(merged_dir),
                split=args.split,
                output_dir=str(annotation_dir),
                device=args.device,
                llm_provider=args.llm_provider,
                llm_model=args.llm_model,
                reasoning_effort=args.reasoning_effort,
                thinking_level=args.thinking_level,
                save_images=args.save_images,
                units=None,
            )
            annotate_concepts(
                threshold_config,
                annotator_args,
                activation_dir,
                sae_dir,
                merged_dir / "concept_names_merged.csv",
                device,
            )
            train_count, train_agreement = read_annotation_stats(annotation_dir / "annotation_summary_train.csv")
            test_count, test_agreement = read_annotation_stats(annotation_dir / "annotation_summary_test.csv")
            gc.collect()
            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

        row = {
            "merge_similarity_threshold": threshold,
            "raw_concepts": len(raw_concepts),
            "exact_unique_concepts": len(intermediate),
            "merged_concepts": len(merged),
            "multi_name_clusters": len(clusters),
            "max_units_per_concept": max_units,
            "config_path": str(config_path),
            "merged_dir": str(merged_dir),
            "annotation_dir": str(annotation_dir) if annotation_dir is not None else "",
            "train_annotated_concepts": train_count,
            "train_mean_llm_sae_agreement": train_agreement,
            "test_annotated_concepts": test_count,
            "test_mean_llm_sae_agreement": test_agreement,
        }
        summary_rows.append(row)
        write_text_atomic(sweep_dir / "sweep_summary.json", json.dumps(summary_rows, indent=2))
        save_csv_atomic(sweep_dir / "sweep_summary.csv", SUMMARY_FIELDNAMES, summary_rows)

    print_summary_table(summary_rows)
    print(f"\nSweep outputs saved under {sweep_dir}")
    print(f"- Pairwise similarities: {sweep_dir / 'similarity_pairs.csv'}")
    print(f"- Cluster merge details: {sweep_dir / 'cluster_details.txt'}")
    print(f"- Summary table: {sweep_dir / 'sweep_summary.csv'}")


def main():
    args = parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()
