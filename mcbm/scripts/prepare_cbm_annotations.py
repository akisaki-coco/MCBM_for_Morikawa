import argparse
import copy
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import torch

from mcbm.scripts.train_cbm import (
    read_annotation_labels,
    read_concepts_from_annotations,
    validate_annotation_pair,
)


SUMMARY_FIELDNAMES = ["concept_id", "concept_name", "llm_sae_agreement", "n_samples"]

MAPPING_FIELDNAMES = [
    "new_concept_id",
    "orig_concept_id",
    "concept_name",
    "status",
    "drop_reasons",
    "train_n",
    "train_pos",
    "train_neg",
    "train_pos_rate",
    "train_agreement",
    "test_n",
    "test_pos",
    "test_neg",
    "test_pos_rate",
    "test_agreement",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Align, filter, and renumber annotations_train.pt/annotations_test.pt for train_cbm.py."
    )
    parser.add_argument(
        "--annotation-dir",
        required=True,
        help="Input annotation directory containing annotations_train.pt and annotations_test.pt.",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for CBM-ready annotations. Defaults to <annotation_dir>_cbm_ready.",
    )
    parser.add_argument(
        "--min-train-class-count",
        type=int,
        default=2,
        help="Minimum positive and negative labels required in train split (0 to disable, default: 2).",
    )
    parser.add_argument(
        "--min-test-class-count",
        type=int,
        default=0,
        help="Minimum positive and negative labels required in test split (0 to disable, default: 0).",
    )
    parser.add_argument(
        "--min-train-agreement",
        type=float,
        help="Minimum train llm_sae_agreement required to keep a concept.",
    )
    parser.add_argument(
        "--min-test-agreement",
        type=float,
        help="Minimum test llm_sae_agreement required to keep a concept.",
    )
    parser.add_argument(
        "--exclude-names",
        help="Comma-separated exact concept names to exclude (case-insensitive).",
    )
    parser.add_argument(
        "--exclude-substrings",
        help="Comma-separated substrings to exclude if found in concept name (case-insensitive).",
    )
    parser.add_argument(
        "--exclude-ids",
        help="Comma-separated original concept_ids to exclude.",
    )
    parser.add_argument(
        "--drop-viz-leaks",
        action="store_true",
        help="Exclude concepts containing 'highlighted' (visualization prompt leak).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect concept label statistics and filtering results without writing files.",
    )
    return parser.parse_args()


def parse_csv_tokens(value):
    if not value:
        return []
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def parse_exclude_ids(value):
    if not value:
        return set()
    ids = set()
    for item in value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        try:
            ids.add(int(stripped))
        except ValueError as exc:
            raise ValueError(f"Invalid concept id in --exclude-ids: {stripped!r}") from exc
    return ids


def validate_args(args):
    if args.min_train_class_count < 0:
        raise ValueError("--min-train-class-count must be >= 0.")
    if args.min_test_class_count < 0:
        raise ValueError("--min-test-class-count must be >= 0.")
    for flag_name, value in (
        ("--min-train-agreement", args.min_train_agreement),
        ("--min-test-agreement", args.min_test_agreement),
    ):
        if value is not None and (not math.isfinite(value) or value < 0.0 or value > 1.0):
            raise ValueError(f"{flag_name} must be in [0, 1], got {value}.")

    annotation_dir = Path(args.annotation_dir)
    if not annotation_dir.is_dir():
        raise NotADirectoryError(f"Annotation directory not found: {annotation_dir}")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else annotation_dir.parent / f"{annotation_dir.name}_cbm_ready"
    )
    if not args.dry_run and output_dir.resolve() == annotation_dir.resolve():
        raise ValueError("--output-dir cannot be the same as --annotation-dir to avoid overwriting source files.")

    return annotation_dir, output_dir


def load_annotations_pt(path):
    if not path.exists():
        raise FileNotFoundError(f"Required annotation file not found: {path}")
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a dictionary keyed by concept_id.")
    normalized = {}
    for raw_id, detail in data.items():
        concept_id = int(raw_id)
        if concept_id in normalized:
            raise ValueError(f"Duplicate concept_id {concept_id} in {path}.")
        normalized[concept_id] = detail
    return normalized


def load_summary_csv(path, required=False):
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required summary CSV not found: {path}")
        return None
    rows = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            concept_id = int(row["concept_id"])
            rows[concept_id] = {
                "concept_id": concept_id,
                "concept_name": row["concept_name"].strip(),
                "llm_sae_agreement": float(row["llm_sae_agreement"]),
                "n_samples": int(row["n_samples"]),
            }
    return rows


def inspect_split_detail(detail):
    """Validate an annotation detail entry and compute positive/negative counts."""
    if not isinstance(detail, dict) or "images" not in detail or "labels" not in detail:
        return False, "", 0, 0, 0, None

    images = detail["images"]
    labels = detail["labels"]
    if not isinstance(images, list) or not isinstance(labels, list) or len(images) != len(labels):
        return False, str(detail.get("concept") or "").strip(), 0, 0, 0, None

    pos = 0
    neg = 0
    for raw_label in labels:
        if raw_label is None:
            continue
        if isinstance(raw_label, bool):
            return False, str(detail.get("concept") or "").strip(), 0, 0, 0, None
        try:
            val = int(raw_label)
        except (TypeError, ValueError):
            return False, str(detail.get("concept") or "").strip(), 0, 0, 0, None
        if val == 1:
            pos += 1
        elif val == 0:
            neg += 1
        else:
            return False, str(detail.get("concept") or "").strip(), 0, 0, 0, None

    n_valid = pos + neg
    pos_rate = (pos / n_valid) if n_valid > 0 else None
    concept_name = str(detail.get("concept") or "").strip()
    return True, concept_name, n_valid, pos, neg, pos_rate


def evaluate_concepts(
    train_data,
    test_data,
    train_summary,
    test_summary,
    args,
    exclude_names,
    exclude_substrings,
    exclude_ids,
):
    """Evaluate every concept_id across train/test splits and determine kept vs dropped."""
    all_ids = sorted(set(train_data) | set(test_data))
    records = []

    for orig_id in all_ids:
        reasons = []
        in_train = orig_id in train_data
        in_test = orig_id in test_data
        if not in_train:
            reasons.append("missing_in_train")
        if not in_test:
            reasons.append("missing_in_test")

        train_valid, train_name, train_n, train_pos, train_neg, train_pos_rate = (
            inspect_split_detail(train_data[orig_id]) if in_train else (False, "", 0, 0, 0, None)
        )
        test_valid, test_name, test_n, test_pos, test_neg, test_pos_rate = (
            inspect_split_detail(test_data[orig_id]) if in_test else (False, "", 0, 0, 0, None)
        )

        if (in_train and not train_valid) or (in_test and not test_valid):
            reasons.append("invalid_label_length")

        if (in_train and not train_name) or (in_test and not test_name):
            reasons.append("empty_concept_name")
        elif in_train and in_test and train_name != test_name:
            reasons.append("name_mismatch")

        concept_name = train_name or test_name
        candidate_names_lower = {name.lower() for name in (train_name, test_name) if name}

        if in_train and train_valid and args.min_train_class_count > 0:
            if train_pos < args.min_train_class_count:
                reasons.append(f"train_pos_lt_{args.min_train_class_count}")
            if train_neg < args.min_train_class_count:
                reasons.append(f"train_neg_lt_{args.min_train_class_count}")

        if in_test and test_valid and args.min_test_class_count > 0:
            if test_pos < args.min_test_class_count:
                reasons.append(f"test_pos_lt_{args.min_test_class_count}")
            if test_neg < args.min_test_class_count:
                reasons.append(f"test_neg_lt_{args.min_test_class_count}")

        train_agreement = (
            train_summary[orig_id]["llm_sae_agreement"]
            if (train_summary is not None and orig_id in train_summary)
            else None
        )
        test_agreement = (
            test_summary[orig_id]["llm_sae_agreement"]
            if (test_summary is not None and orig_id in test_summary)
            else None
        )

        if args.min_train_agreement is not None:
            if train_agreement is None or train_agreement < args.min_train_agreement:
                reasons.append(f"train_agreement_lt_{args.min_train_agreement:g}")

        if args.min_test_agreement is not None:
            if test_agreement is None or test_agreement < args.min_test_agreement:
                reasons.append(f"test_agreement_lt_{args.min_test_agreement:g}")

        if orig_id in exclude_ids:
            reasons.append("excluded_id")
        if any(name in exclude_names for name in candidate_names_lower):
            reasons.append("excluded_name")
        if any(sub in name for name in candidate_names_lower for sub in exclude_substrings):
            reasons.append("excluded_substring")

        records.append(
            {
                "orig_concept_id": orig_id,
                "new_concept_id": None,
                "concept_name": concept_name,
                "status": "kept" if not reasons else "dropped",
                "drop_reasons": reasons,
                "train_n": train_n,
                "train_pos": train_pos,
                "train_neg": train_neg,
                "train_pos_rate": train_pos_rate,
                "train_agreement": train_agreement,
                "test_n": test_n,
                "test_pos": test_pos,
                "test_neg": test_neg,
                "test_pos_rate": test_pos_rate,
                "test_agreement": test_agreement,
            }
        )

    kept_index = 0
    for record in records:
        if record["status"] == "kept":
            record["new_concept_id"] = kept_index
            kept_index += 1

    return records


def print_diagnostic_table(records):
    print("\n=== Concept Annotation Diagnostic Table ===")
    header = (
        f"{'orig':>4} | {'new':>3} | {'status':<7} | {'tr_pos/n (rate)':<16} | {'tr_agr':>6} | "
        f"{'te_pos/n (rate)':<16} | {'te_agr':>6} | {'concept_name':<34} | reasons"
    )
    print(header)
    print("-" * len(header))
    for rec in records:
        new_str = str(rec["new_concept_id"]) if rec["new_concept_id"] is not None else "-"
        tr_rate = f"{rec['train_pos_rate']:.2f}" if rec["train_pos_rate"] is not None else " -  "
        te_rate = f"{rec['test_pos_rate']:.2f}" if rec["test_pos_rate"] is not None else " -  "
        tr_info = f"{rec['train_pos']:>3}/{rec['train_n']:<3} ({tr_rate})"
        te_info = f"{rec['test_pos']:>3}/{rec['test_n']:<3} ({te_rate})"
        tr_agr = f"{rec['train_agreement']:.3f}" if rec["train_agreement"] is not None else "  -   "
        te_agr = f"{rec['test_agreement']:.3f}" if rec["test_agreement"] is not None else "  -   "
        reasons_str = ",".join(rec["drop_reasons"])
        print(
            f"{rec['orig_concept_id']:>4} | {new_str:>3} | {rec['status']:<7} | "
            f"{tr_info:<16} | {tr_agr:>6} | {te_info:<16} | {te_agr:>6} | "
            f"{rec['concept_name'][:34]:<34} | {reasons_str}"
        )

    kept_count = sum(1 for r in records if r["status"] == "kept")
    dropped_count = len(records) - kept_count
    print(f"\nTotal candidate concepts: {len(records)} | Kept: {kept_count} | Dropped: {dropped_count}")


def write_csv_atomic(path, fieldnames, rows):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp_path.replace(path)


def write_text_atomic(path, text):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text)
    tmp_path.replace(path)


def save_outputs(
    annotation_dir,
    output_dir,
    records,
    train_data,
    test_data,
    train_summary,
    test_summary,
    args,
    exclude_names,
    exclude_substrings,
    exclude_ids,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    kept_records = [rec for rec in records if rec["status"] == "kept"]

    new_train = {}
    new_test = {}
    new_train_summary = []
    new_test_summary = []

    for rec in kept_records:
        orig_id = rec["orig_concept_id"]
        new_id = rec["new_concept_id"]
        train_entry = copy.deepcopy(train_data[orig_id])
        train_entry["concept"] = rec["concept_name"]
        new_train[new_id] = train_entry

        test_entry = copy.deepcopy(test_data[orig_id])
        test_entry["concept"] = rec["concept_name"]
        new_test[new_id] = test_entry

        if train_summary is not None and orig_id in train_summary:
            row = dict(train_summary[orig_id])
            row["concept_id"] = new_id
            new_train_summary.append(row)
        if test_summary is not None and orig_id in test_summary:
            row = dict(test_summary[orig_id])
            row["concept_id"] = new_id
            new_test_summary.append(row)

    out_train_pt = output_dir / "annotations_train.pt"
    out_test_pt = output_dir / "annotations_test.pt"
    torch.save(new_train, out_train_pt)
    torch.save(new_test, out_test_pt)

    if train_summary is not None:
        write_csv_atomic(output_dir / "annotation_summary_train.csv", SUMMARY_FIELDNAMES, new_train_summary)
    if test_summary is not None:
        write_csv_atomic(output_dir / "annotation_summary_test.csv", SUMMARY_FIELDNAMES, new_test_summary)

    mapping_rows = [
        {
            "new_concept_id": rec["new_concept_id"] if rec["new_concept_id"] is not None else "",
            "orig_concept_id": rec["orig_concept_id"],
            "concept_name": rec["concept_name"],
            "status": rec["status"],
            "drop_reasons": ";".join(rec["drop_reasons"]),
            "train_n": rec["train_n"],
            "train_pos": rec["train_pos"],
            "train_neg": rec["train_neg"],
            "train_pos_rate": rec["train_pos_rate"] if rec["train_pos_rate"] is not None else "",
            "train_agreement": rec["train_agreement"] if rec["train_agreement"] is not None else "",
            "test_n": rec["test_n"],
            "test_pos": rec["test_pos"],
            "test_neg": rec["test_neg"],
            "test_pos_rate": rec["test_pos_rate"] if rec["test_pos_rate"] is not None else "",
            "test_agreement": rec["test_agreement"] if rec["test_agreement"] is not None else "",
        }
        for rec in records
    ]
    write_csv_atomic(output_dir / "concept_mapping.csv", MAPPING_FIELDNAMES, mapping_rows)

    run_info_path = annotation_dir / "run_info.json"
    run_info = json.loads(run_info_path.read_text()) if run_info_path.exists() else {}
    run_info["source_annotation_dir"] = str(annotation_dir)
    run_info["preprocessing"] = {
        "min_train_class_count": args.min_train_class_count,
        "min_test_class_count": args.min_test_class_count,
        "min_train_agreement": args.min_train_agreement,
        "min_test_agreement": args.min_test_agreement,
        "exclude_names": sorted(exclude_names),
        "exclude_substrings": sorted(exclude_substrings),
        "exclude_ids": sorted(exclude_ids),
        "drop_viz_leaks": bool(args.drop_viz_leaks),
        "total_candidate_concepts": len(records),
        "kept_concepts": len(kept_records),
        "dropped_concepts": len(records) - len(kept_records),
        "created_at": datetime.now().isoformat(),
    }
    write_text_atomic(output_dir / "run_info.json", json.dumps(run_info, indent=2))

    # Self-verify compatibility with train_cbm.py loaders and validators.
    verified_concepts = read_concepts_from_annotations(out_train_pt)
    validate_annotation_pair(out_train_pt, out_test_pt, verified_concepts)
    read_annotation_labels(out_train_pt)
    read_annotation_labels(out_test_pt)

    return run_info, len(verified_concepts)


def main():
    args = parse_args()
    annotation_dir, output_dir = validate_args(args)

    exclude_names = set(parse_csv_tokens(args.exclude_names))
    exclude_substrings = set(parse_csv_tokens(args.exclude_substrings))
    if args.drop_viz_leaks:
        exclude_substrings.add("highlighted")
    exclude_ids = parse_exclude_ids(args.exclude_ids)

    train_data = load_annotations_pt(annotation_dir / "annotations_train.pt")
    test_data = load_annotations_pt(annotation_dir / "annotations_test.pt")
    train_summary = load_summary_csv(
        annotation_dir / "annotation_summary_train.csv",
        required=(args.min_train_agreement is not None),
    )
    test_summary = load_summary_csv(
        annotation_dir / "annotation_summary_test.csv",
        required=(args.min_test_agreement is not None),
    )

    records = evaluate_concepts(
        train_data,
        test_data,
        train_summary,
        test_summary,
        args,
        exclude_names,
        exclude_substrings,
        exclude_ids,
    )
    print_diagnostic_table(records)

    kept_count = sum(1 for rec in records if rec["status"] == "kept")
    if args.dry_run:
        if kept_count == 0:
            print("Warning: No concepts remain after filtering (dry-run).")
        else:
            print(f"Dry-run complete. Would save {kept_count} concepts to {output_dir}.")
        return

    if kept_count == 0:
        raise ValueError("No concepts remain after filtering. Relax filter criteria and try again.")

    run_info, verified_count = save_outputs(
        annotation_dir,
        output_dir,
        records,
        train_data,
        test_data,
        train_summary,
        test_summary,
        args,
        exclude_names,
        exclude_substrings,
        exclude_ids,
    )

    print(f"\nSaved {verified_count} CBM-ready concepts to {output_dir} (verified against train_cbm.py).")
    cfg_arg = run_info.get("config") or "<CONFIG_PATH>"
    act_arg = run_info.get("activation_dir") or "<ACTIVATION_DIR>"
    print("\nNext step:")
    print(
        f"  python -m mcbm.scripts.train_cbm --config {cfg_arg} "
        f"--activation-dir {act_arg} --annotation-dir {output_dir}"
    )


if __name__ == "__main__":
    main()
