import argparse
import copy
import csv
import gc
import json
import math
import re
from datetime import datetime
from pathlib import Path

import torch

from mcbm.scripts.annotate_concepts import annotate_concepts, get_annotation_budget
from mcbm.scripts.merge_concepts import (
    get_config_float,
    load_run_info,
    merge_concepts,
    merge_identical_names,
    read_concepts,
    save_merged_output,
)
from mcbm.scripts.name_concepts import (
    load_kept_indices,
    name_concepts,
    parse_units,
    read_existing_names,
    resolve_api_key,
    validate_llm_options,
)
from mcbm.scripts.train_sae import require_sae_config, train_sae


DEFAULT_PARAM_KEY = "sae_expansion_factor"
DEFAULT_EXPANSION_FACTORS = "0.25,0.5,1,2,4,8"
DEFAULT_LLM_MODEL = "qwen3.5:27b"
DEFAULT_EMBEDDING_MODEL = "qwen3-embedding:0.6b"

BASE_SUMMARY_TAIL_FIELDNAMES = [
    "input_dim",
    "latent_dim",
    "kept_units",
    "pruned_units",
    "named_concepts",
    "merged_concepts",
    "val_recovered_loss",
    "val_recovered_accuracy",
    "selected_recovered_loss",
    "selected_recovered_accuracy",
    "train_annotated_concepts",
    "train_mean_llm_sae_agreement",
    "test_annotated_concepts",
    "test_mean_llm_sae_agreement",
    "config_path",
    "sae_dir",
    "concept_dir",
    "annotation_dir",
]

SUMMARY_FIELDNAMES = [DEFAULT_PARAM_KEY] + BASE_SUMMARY_TAIL_FIELDNAMES


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep a config parameter through SAE training, concept naming, merging, and annotation."
    )
    parser.add_argument("--config", required=True, help="Path to base dataset config JSON.")
    parser.add_argument("--activation-dir", required=True, help="Directory containing saved activation files.")
    parser.add_argument(
        "--param-key",
        default=DEFAULT_PARAM_KEY,
        help=f"Config JSON key to sweep (default: {DEFAULT_PARAM_KEY}).",
    )
    parser.add_argument(
        "--param-values",
        default=None,
        help="Comma-separated values to sweep for --param-key.",
    )
    parser.add_argument(
        "--expansion-factors",
        default=None,
        help=(
            f"Backward-compatible alias for --param-values when --param-key is {DEFAULT_PARAM_KEY} "
            f"(default values when omitted: {DEFAULT_EXPANSION_FACTORS})."
        ),
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for sweep configs and summary outputs. Defaults to outputs/sweeps/<dataset>/sweep_<param>_<timestamp>.",
    )
    parser.add_argument("--device", help="Torch device override, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument(
        "--llm-provider",
        choices=["openai", "gemini", "ollama"],
        default="ollama",
        help="LLM provider for concept naming and annotation (default: ollama).",
    )
    parser.add_argument(
        "--llm-model",
        default=DEFAULT_LLM_MODEL,
        help=f"LLM model name for concept naming and annotation (default: {DEFAULT_LLM_MODEL}).",
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
        "--split",
        choices=["all", "train", "test"],
        default="all",
        help="Annotation split (default: all).",
    )
    parser.add_argument("--reasoning-effort", choices=["none", "minimal", "low", "medium", "high", "xhigh"])
    parser.add_argument("--thinking-level", choices=["minimal", "low", "medium", "high"])
    parser.add_argument("--save-images", action="store_true", help="Save naming and annotation debug images.")
    parser.add_argument("--units", help="Comma-separated post-pruning unit ids to name, e.g. 1,2,3.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing completed/in-progress SAE, concept, and annotation directories for each sweep value.",
    )
    return parser.parse_args()


def _parse_numeric_item(item, param_key):
    try:
        numeric = float(item)
    except ValueError as exc:
        raise ValueError(f"Invalid value '{item}' for {param_key}: must be a number.") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"Value for {param_key} must be finite, got {item}.")
    return int(numeric) if numeric.is_integer() else numeric


def _parse_single_param_value(item, param_key, base_value):
    if isinstance(base_value, bool):
        lowered = item.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        raise ValueError(f"Invalid boolean value '{item}' for {param_key}; expected 'true' or 'false'.")

    if isinstance(base_value, (int, float)):
        return _parse_numeric_item(item, param_key)

    if base_value is None:
        lowered = item.lower()
        if lowered in ("none", "null"):
            return None
        try:
            return _parse_numeric_item(item, param_key)
        except ValueError:
            return item

    if isinstance(base_value, str):
        return item

    raise TypeError(f"Unsupported base config value type for {param_key}: {type(base_value).__name__}")


def parse_param_values(raw_value, param_key, base_config):
    """Parse and validate a comma-separated list of values matching base_config[param_key]'s type."""
    if param_key not in base_config:
        raise KeyError(f"Config is missing target sweep key: {param_key}")

    items = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not items:
        raise ValueError(f"Sweep values for {param_key} must contain at least one value.")

    base_value = base_config[param_key]
    values = []
    seen = set()
    for item in items:
        parsed = _parse_single_param_value(item, param_key, base_value)
        dedup_key = (isinstance(parsed, bool), parsed)
        if dedup_key in seen:
            raise ValueError(f"Duplicate value detected for {param_key}: {parsed}")
        seen.add(dedup_key)
        values.append(parsed)
    return values


def resolve_sweep_values(args, base_config):
    """Resolve CLI sweep parameter key and values with backward compatibility."""
    param_key = args.param_key
    if args.param_values is not None and args.expansion_factors is not None:
        raise ValueError("Specify either --param-values or --expansion-factors, not both.")
    if param_key != DEFAULT_PARAM_KEY and args.expansion_factors is not None:
        raise ValueError("--expansion-factors can only be used when --param-key is sae_expansion_factor.")
    if param_key != DEFAULT_PARAM_KEY and args.param_values is None:
        raise ValueError(f"--param-values is required when --param-key is {param_key}.")

    raw_value = (
        args.param_values
        if args.param_values is not None
        else (args.expansion_factors if args.expansion_factors is not None else DEFAULT_EXPANSION_FACTORS)
    )
    return param_key, parse_param_values(raw_value, param_key, base_config)


def parse_expansion_factors(value):
    """Backward-compatible helper to parse positive sae_expansion_factor values."""
    factors = parse_param_values(value, DEFAULT_PARAM_KEY, {DEFAULT_PARAM_KEY: 1.0})
    for factor in factors:
        if factor <= 0:
            raise ValueError(f"Expansion factor must be positive, got {factor}.")
    return factors


def format_param_prefix(param_key):
    if param_key == DEFAULT_PARAM_KEY:
        return "ef"
    return re.sub(r"[^A-Za-z0-9]+", "_", param_key).strip("_") or "param"


def format_value_tag(value):
    """Format a sweep parameter value as a filesystem-safe tag."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "none"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:g}".replace(".", "_").replace("-", "m").replace("+", "")
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_") or "val"


def format_factor_tag(factor):
    """Backward-compatible formatter for sae_expansion_factor tags."""
    return format_value_tag(factor)


def build_run_tag(param_key, value):
    val_tag = format_value_tag(value)
    if param_key == DEFAULT_PARAM_KEY:
        return f"ef{val_tag}"
    return f"{format_param_prefix(param_key)}_{val_tag}"


def validate_activation_dir(activation_dir):
    """Ensure required activation split files exist before starting the sweep."""
    activation_dir = Path(activation_dir)
    if not activation_dir.exists():
        raise FileNotFoundError(f"Activation directory not found: {activation_dir}")
    for split in ("train", "val", "test"):
        split_path = activation_dir / f"activations_{split}.pt"
        if not split_path.exists():
            raise FileNotFoundError(f"Required activation file not found: {split_path}")
    return activation_dir


def validate_config_values(args, candidate_config):
    """Validate all SAE, merge, and annotation parameters on a candidate config."""
    require_sae_config(candidate_config)

    for key, allow_zero in (
        ("sae_expansion_factor", False),
        ("sae_l1_coefficient", True),
        ("sae_learning_rate", False),
        ("sae_ce_drop_tolerance", True),
    ):
        val = candidate_config[key]
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(float(val)):
            raise ValueError(f"{key} must be a finite number, got {val!r}.")
        if allow_zero:
            if float(val) < 0:
                raise ValueError(f"{key} must be >= 0, got {val}.")
        elif float(val) <= 0:
            raise ValueError(f"{key} must be > 0, got {val}.")

    for int_key in ("sae_batch_size", "sae_epochs", "sae_patience", "sae_min_active_images"):
        val = candidate_config[int_key]
        if isinstance(val, bool) or not isinstance(val, int) or val < 1:
            raise ValueError(f"{int_key} must be an integer >= 1, got {val!r}.")

    threshold = get_config_float(candidate_config, "merge_similarity_threshold")
    if not math.isfinite(threshold) or threshold <= 0.0 or threshold > 1.0:
        raise ValueError(f"merge_similarity_threshold must be finite and in (0, 1], got {threshold}.")

    if args.split in ("all", "train"):
        get_annotation_budget(candidate_config, "train")
    if args.split in ("all", "test"):
        get_annotation_budget(candidate_config, "test")


def validate_sweep_options(args, base_config, candidate_configs=None):
    """Validate CLI flags, provider models, API keys, and base/candidate configs upfront."""
    validate_llm_options(args)
    if args.llm_provider in ("openai", "gemini"):
        if args.llm_model == DEFAULT_LLM_MODEL:
            raise ValueError(f"Specify --llm-model when using --llm-provider {args.llm_provider}.")
        resolve_api_key(args.llm_provider)

    if args.embedding_provider in ("openai", "gemini"):
        if args.embedding_model == DEFAULT_EMBEDDING_MODEL:
            raise ValueError(f"Specify --embedding-model when using --embedding-provider {args.embedding_provider}.")
        resolve_api_key(args.embedding_provider)

    configs_to_check = candidate_configs if candidate_configs is not None else [base_config]
    for candidate_config in configs_to_check:
        validate_config_values(args, candidate_config)


def sanitize_model_name(provider, model):
    return f"{provider}_{model}".replace(".", "_").replace("-", "_").replace(":", "_")


def build_step_dirs(config, args, run_tag):
    """Construct timestamped output directories tagged with the sweep parameter."""
    dataset = config["dataset"]
    backbone = config["backbone"]
    feature_layer = config["feature_layer"].replace(".", "_")
    model_name = sanitize_model_name(args.llm_provider, args.llm_model)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    sae_dir = Path("outputs") / "sae" / dataset / f"sae_{dataset}_{backbone}_{feature_layer}_{run_tag}_{timestamp}"
    concept_dir = Path("outputs") / "concepts" / dataset / f"concepts_{dataset}_{model_name}_{run_tag}_{timestamp}"
    annotation_dir = (
        Path("outputs") / "annotations" / dataset / f"annotations_{dataset}_{model_name}_{run_tag}_{timestamp}"
    )
    return sae_dir, concept_dir, annotation_dir


def _paths_match(path_a, path_b):
    if not path_a or not path_b:
        return False
    return Path(path_a).resolve() == Path(path_b).resolve()


def _values_match(saved_value, expected_value):
    if isinstance(saved_value, bool) or isinstance(expected_value, bool):
        return type(saved_value) is type(expected_value) and saved_value == expected_value
    if isinstance(saved_value, (int, float)) and isinstance(expected_value, (int, float)):
        return math.isclose(float(saved_value), float(expected_value), rel_tol=1e-9, abs_tol=1e-12)
    return saved_value == expected_value


def _read_run_info_if_valid(run_dir):
    path = Path(run_dir) / "run_info.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _find_latest_matching_dir(parent, pattern, run_info_key, expected_path, predicate=None):
    if not parent.exists():
        return None
    for candidate in sorted(parent.glob(pattern), key=lambda p: p.name, reverse=True):
        if not candidate.is_dir():
            continue
        if predicate is not None and not predicate(candidate):
            continue
        run_info = _read_run_info_if_valid(candidate)
        if run_info is not None and _paths_match(run_info.get(run_info_key), expected_path):
            return candidate
    return None


def _is_completed_sae_dir(candidate):
    return (
        (candidate / "metrics.json").exists()
        and (candidate / "pruning" / "kept_indices.pt").exists()
        and len(list(candidate.glob("*.pt"))) == 1
    )


def find_resumable_sae_dir(config, run_tag, activation_dir, param_key=None, param_value=None):
    """Return the newest completed SAE directory matching this run_tag, activation_dir, and parameter."""
    dataset = config["dataset"]
    backbone = config["backbone"]
    feature_layer = config["feature_layer"].replace(".", "_")
    parent = Path("outputs") / "sae" / dataset
    if not parent.exists():
        return None

    pattern = f"sae_{dataset}_{backbone}_{feature_layer}_{run_tag}_*"
    for candidate in sorted(parent.glob(pattern), key=lambda p: p.name, reverse=True):
        if not candidate.is_dir() or not _is_completed_sae_dir(candidate):
            continue
        run_info = _read_run_info_if_valid(candidate)
        if run_info is None or not _paths_match(run_info.get("activation_dir"), activation_dir):
            continue
        if param_key is not None:
            params = run_info.get("parameters")
            if isinstance(params, dict) and param_key in params:
                if not _values_match(params[param_key], param_value):
                    continue
        return candidate
    return None


def find_resumable_concept_dir(config, args, run_tag, sae_dir):
    """Return the newest concept directory matching this run_tag and sae_dir."""
    dataset = config["dataset"]
    model_name = sanitize_model_name(args.llm_provider, args.llm_model)
    return _find_latest_matching_dir(
        Path("outputs") / "concepts" / dataset,
        f"concepts_{dataset}_{model_name}_{run_tag}_*",
        "sae_dir",
        sae_dir,
    )


def is_concept_naming_complete(sae_dir, concept_dir, units_arg):
    """Check whether all target post-pruning units already appear in concept_names.csv."""
    names_path = Path(concept_dir) / "concept_names.csv"
    if not names_path.exists():
        return False

    kept_indices = load_kept_indices(sae_dir)
    requested_units = parse_units(units_arg)
    if requested_units is not None:
        available_units = set(range(int(kept_indices.numel())))
        missing_units = sorted(requested_units - available_units)
        if missing_units:
            raise ValueError(f"Requested units are outside the post-pruning range: {missing_units}")
        expected_units = requested_units
    else:
        expected_units = set(range(int(kept_indices.numel())))

    existing_names = read_existing_names(names_path)
    return expected_units.issubset(existing_names.keys())


def find_resumable_annotation_dir(config, args, run_tag, concept_dir):
    """Return the newest annotation directory matching this run_tag and concept_dir."""
    dataset = config["dataset"]
    model_name = sanitize_model_name(args.llm_provider, args.llm_model)
    return _find_latest_matching_dir(
        Path("outputs") / "annotations" / dataset,
        f"annotations_{dataset}_{model_name}_{run_tag}_*",
        "concept_dir",
        concept_dir,
    )


def run_merge_with_fallback(factor_config, concept_dir, embedding_provider, embedding_model):
    """Run concept merging, falling back cleanly if only one unique concept name exists."""
    concepts = read_concepts(concept_dir / "concept_names.csv")
    intermediate = merge_identical_names(concepts)
    if len(intermediate) == 1:
        print("Only one unique concept name remains after exact-name merging; saving directly without clustering.")
        run_info = load_run_info(concept_dir)
        run_info["embedding_provider"] = embedding_provider
        run_info["embedding_model"] = embedding_model
        save_merged_output(concept_dir, concept_dir, run_info, intermediate)
        return

    merge_concepts(
        factor_config,
        concept_dir,
        concept_dir,
        embedding_provider,
        embedding_model,
    )


def count_csv_rows(path):
    if not path.exists():
        return 0
    with path.open(newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def read_annotation_stats(summary_path):
    """Read annotated concept count and mean agreement from an annotation summary CSV."""
    if not summary_path.exists():
        return 0, None

    with summary_path.open(newline="") as f:
        agreements = [float(row["llm_sae_agreement"]) for row in csv.DictReader(f)]

    if not agreements:
        return 0, None
    return len(agreements), sum(agreements) / len(agreements)


def get_summary_fieldnames(param_key):
    if param_key == DEFAULT_PARAM_KEY:
        return [DEFAULT_PARAM_KEY] + BASE_SUMMARY_TAIL_FIELDNAMES
    return [param_key, DEFAULT_PARAM_KEY] + BASE_SUMMARY_TAIL_FIELDNAMES


def collect_run_summary(param_key, param_value, run_config, run_config_path, sae_dir, concept_dir, annotation_dir):
    """Collect key metrics from SAE, concept merging, and annotation outputs."""
    metrics = json.loads((sae_dir / "metrics.json").read_text())
    pruning = metrics["pruning"]
    train_count, train_agreement = read_annotation_stats(annotation_dir / "annotation_summary_train.csv")
    test_count, test_agreement = read_annotation_stats(annotation_dir / "annotation_summary_test.csv")

    row = {
        param_key: param_value,
        "input_dim": metrics["input_dim"],
        "latent_dim": metrics["latent_dim"],
        "kept_units": pruning["kept_units"],
        "pruned_units": pruning["pruned_units"],
        "named_concepts": count_csv_rows(concept_dir / "concept_names.csv"),
        "merged_concepts": count_csv_rows(concept_dir / "concept_names_merged.csv"),
        "val_recovered_loss": metrics["val_recovered_loss"],
        "val_recovered_accuracy": metrics["val_recovered_accuracy"],
        "selected_recovered_loss": pruning["selected_recovered_loss"],
        "selected_recovered_accuracy": pruning["selected_recovered_accuracy"],
        "train_annotated_concepts": train_count,
        "train_mean_llm_sae_agreement": train_agreement,
        "test_annotated_concepts": test_count,
        "test_mean_llm_sae_agreement": test_agreement,
        "config_path": str(run_config_path),
        "sae_dir": str(sae_dir),
        "concept_dir": str(concept_dir),
        "annotation_dir": str(annotation_dir),
    }
    if param_key != DEFAULT_PARAM_KEY:
        row[DEFAULT_PARAM_KEY] = run_config[DEFAULT_PARAM_KEY]
    return row


def collect_factor_summary(factor, factor_config_path, sae_dir, concept_dir, annotation_dir):
    """Backward-compatible wrapper for sae_expansion_factor summary collection."""
    return collect_run_summary(
        DEFAULT_PARAM_KEY,
        factor,
        {DEFAULT_PARAM_KEY: factor},
        factor_config_path,
        sae_dir,
        concept_dir,
        annotation_dir,
    )


def save_sweep_summary(sweep_dir, rows, fieldnames=None):
    """Write sweep_summary.json and sweep_summary.csv atomically after each iteration."""
    fieldnames = fieldnames if fieldnames is not None else SUMMARY_FIELDNAMES
    json_path = sweep_dir / "sweep_summary.json"
    tmp_json = json_path.with_suffix(".tmp")
    tmp_json.write_text(json.dumps(rows, indent=2))
    tmp_json.replace(json_path)

    csv_path = sweep_dir / "sweep_summary.csv"
    tmp_csv = csv_path.with_suffix(".tmp")
    with tmp_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp_csv.replace(csv_path)


def prepare_sweep_dir(base_config, output_dir, param_key=DEFAULT_PARAM_KEY):
    if output_dir is not None:
        return Path(output_dir)
    dataset = base_config["dataset"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = format_param_prefix(param_key)
    return Path("outputs") / "sweeps" / dataset / f"sweep_{prefix}_{timestamp}"


def _cleanup_memory(device):
    gc.collect()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_sweep(args):
    base_config = json.loads(Path(args.config).read_text())
    param_key, param_values = resolve_sweep_values(args, base_config)
    activation_dir = validate_activation_dir(args.activation_dir)

    candidate_configs = []
    for param_value in param_values:
        candidate = copy.deepcopy(base_config)
        candidate[param_key] = param_value
        candidate_configs.append(candidate)
    validate_sweep_options(args, base_config, candidate_configs=candidate_configs)

    device = torch.device(args.device or base_config.get("device", "cuda"))
    sweep_dir = prepare_sweep_dir(base_config, args.output_dir, param_key=param_key)
    configs_dir = sweep_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    run_info = {
        "base_config": str(args.config),
        "activation_dir": str(activation_dir),
        "param_key": param_key,
        "param_values": param_values,
        "device": str(device),
        "llm_provider": args.llm_provider,
        "llm_model": args.llm_model,
        "embedding_provider": args.embedding_provider,
        "embedding_model": args.embedding_model,
        "split": args.split,
        "reasoning_effort": args.reasoning_effort,
        "thinking_level": args.thinking_level,
        "save_images": bool(args.save_images),
        "units": args.units,
        "resume": bool(args.resume),
        "started_at": datetime.now().isoformat(),
    }
    if param_key == DEFAULT_PARAM_KEY:
        run_info["expansion_factors"] = param_values
    (sweep_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))

    param_prefix = format_param_prefix(param_key)
    summary_fieldnames = get_summary_fieldnames(param_key)
    summary_rows = []

    for index, (param_value, run_config) in enumerate(zip(param_values, candidate_configs), start=1):
        val_tag = format_value_tag(param_value)
        run_tag = build_run_tag(param_key, param_value)

        run_config_path = configs_dir / f"config_{param_prefix}_{val_tag}.json"
        run_config_path.write_text(json.dumps(run_config, indent=4))

        default_sae_dir, default_concept_dir, default_annotation_dir = build_step_dirs(
            run_config, args, run_tag
        )

        resumed_sae = (
            find_resumable_sae_dir(
                run_config,
                run_tag,
                activation_dir,
                param_key=param_key,
                param_value=param_value,
            )
            if args.resume
            else None
        )
        sae_dir = resumed_sae if resumed_sae is not None else default_sae_dir

        resumed_concept = (
            find_resumable_concept_dir(run_config, args, run_tag, sae_dir)
            if (args.resume and resumed_sae is not None)
            else None
        )
        concept_dir = resumed_concept if resumed_concept is not None else default_concept_dir

        resumed_annotation = (
            find_resumable_annotation_dir(run_config, args, run_tag, concept_dir)
            if (args.resume and resumed_concept is not None)
            else None
        )
        annotation_dir = resumed_annotation if resumed_annotation is not None else default_annotation_dir

        print(f"\n[{index}/{len(param_values)}] Starting sweep for {param_key}={param_value}")
        print(f"Config: {run_config_path}")
        print(f"SAE dir: {sae_dir}")
        print(f"Concept dir: {concept_dir}")
        print(f"Annotation dir: {annotation_dir}")

        if resumed_sae is not None:
            print(f"[resume] Reusing completed SAE dir: {sae_dir}")
        else:
            train_sae(run_config, run_config_path, activation_dir, sae_dir, device)
            _cleanup_memory(device)

        if resumed_concept is not None and is_concept_naming_complete(sae_dir, concept_dir, args.units):
            print(f"[resume] Skipping concept naming (all units already named in {concept_dir / 'concept_names.csv'}).")
        else:
            if resumed_concept is not None:
                print(f"[resume] Resuming concept naming in {concept_dir}")
            namer_args = argparse.Namespace(
                config=str(run_config_path),
                sae_dir=str(sae_dir),
                output_dir=str(concept_dir),
                device=args.device,
                save_images=args.save_images,
                llm_provider=args.llm_provider,
                llm_model=args.llm_model,
                reasoning_effort=args.reasoning_effort,
                thinking_level=args.thinking_level,
                units=args.units,
            )
            name_concepts(run_config, namer_args, activation_dir, device)
            _cleanup_memory(device)

        merged_csv_path = concept_dir / "concept_names_merged.csv"
        if args.resume and merged_csv_path.exists():
            print(f"[resume] Skipping concept merging ({merged_csv_path} already exists).")
        else:
            run_merge_with_fallback(
                run_config,
                concept_dir,
                args.embedding_provider,
                args.embedding_model,
            )

        if resumed_annotation is not None:
            print(f"[resume] Resuming annotation in {annotation_dir}")
        # name_concepts filters by post-pruning unit ids, while merge_concepts renumbers
        # merged concepts from 0, so annotate_concepts always annotates all merged concepts.
        annotator_args = argparse.Namespace(
            config=str(run_config_path),
            concept_dir=str(concept_dir),
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
            run_config,
            annotator_args,
            activation_dir,
            sae_dir,
            merged_csv_path,
            device,
        )

        row = collect_run_summary(
            param_key,
            param_value,
            run_config,
            run_config_path,
            sae_dir,
            concept_dir,
            annotation_dir,
        )
        summary_rows.append(row)
        save_sweep_summary(sweep_dir, summary_rows, fieldnames=summary_fieldnames)
        print(
            f"Completed {param_key}={param_value}: "
            f"kept={row['kept_units']}/{row['latent_dim']}, "
            f"merged_concepts={row['merged_concepts']}, "
            f"train_agree={row['train_mean_llm_sae_agreement']}, "
            f"test_agree={row['test_mean_llm_sae_agreement']}"
        )

        _cleanup_memory(device)

    print(f"\nSweep finished. Summary saved to {sweep_dir / 'sweep_summary.csv'}")


def main():
    args = parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()
