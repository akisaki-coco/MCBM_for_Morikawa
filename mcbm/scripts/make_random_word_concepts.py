import argparse
import csv
import itertools
import json
import random
from pathlib import Path


# Everyday concrete nouns clearly unrelated to medical / pathology / cancer domains.
BUILTIN_NOUNS = (
    "accordion",
    "anchor",
    "anvil",
    "apron",
    "backpack",
    "balloon",
    "banjo",
    "barrel",
    "basket",
    "bicycle",
    "binoculars",
    "blanket",
    "blender",
    "bookcase",
    "boomerang",
    "boots",
    "bowling",
    "bracelet",
    "bridge",
    "briefcase",
    "broom",
    "bucket",
    "cabinet",
    "cactus",
    "calculator",
    "calendar",
    "camera",
    "campfire",
    "candle",
    "canoe",
    "carpet",
    "carriage",
    "chair",
    "chandelier",
    "chessboard",
    "chimney",
    "clock",
    "compass",
    "couch",
    "crowbar",
    "curtain",
    "cushion",
    "desk",
    "dice",
    "doormat",
    "drum",
    "dumbbell",
    "envelope",
    "escalator",
    "faucet",
    "fence",
    "fireplace",
    "flashlight",
    "flute",
    "forklift",
    "fountain",
    "freezer",
    "gazebo",
    "glacier",
    "globe",
    "gondola",
    "guitar",
    "hammer",
    "hammock",
    "harmonica",
    "harp",
    "hatchet",
    "helmet",
    "hinge",
    "hourglass",
    "kayak",
    "kettle",
    "keyboard",
    "kite",
    "ladder",
    "lamp",
    "lantern",
    "lawnmower",
    "lighthouse",
    "luggage",
    "mailbox",
    "marimba",
    "megaphone",
    "metronome",
    "microwave",
    "windmill",
    "mitten",
    "monument",
    "motorcycle",
    "mountain",
    "notebook",
    "oar",
    "padlock",
    "paintbrush",
    "parachute",
    "piano",
    "picnic",
    "pillow",
    "pitcher",
    "podium",
    "popcorn",
    "postbox",
    "pretzel",
    "projector",
    "pyramid",
    "quilt",
    "racquet",
    "railroad",
    "refrigerator",
    "sailboat",
    "saxophone",
    "scarecrow",
    "scissors",
    "scooter",
    "screwdriver",
    "seesaw",
    "shovel",
    "skateboard",
    "sled",
    "snowman",
    "sofa",
    "spoon",
    "staircase",
    "statue",
    "stove",
    "submarine",
    "suitcase",
    "surfboard",
    "swing",
    "teapot",
    "telescope",
    "tent",
    "thermometer",
    "toaster",
    "toolbox",
    "tractor",
    "trafficlight",
    "trampoline",
    "treadmill",
    "tricycle",
    "trombone",
    "trophy",
    "trumpet",
    "typewriter",
    "umbrella",
    "violin",
    "volcano",
    "wagon",
    "wallet",
    "wardrobe",
    "wheelbarrow",
    "whistle",
    "wrench",
    "xylophone",
    "yacht",
)

BUILTIN_MODIFIERS = (
    "antique",
    "brass",
    "cardboard",
    "ceramic",
    "checkered",
    "clay",
    "copper",
    "cotton",
    "denim",
    "folded",
    "frosted",
    "glass",
    "granite",
    "knitted",
    "leather",
    "linen",
    "miniature",
    "painted",
    "paper",
    "plastic",
    "polished",
    "rusty",
    "striped",
    "tin",
    "velvet",
    "vintage",
    "wooden",
    "woven",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a random-word control concept_dir from an existing merged concept_dir."
    )
    parser.add_argument(
        "--concept-dir",
        required=True,
        help="Source merged concept directory containing concept_names_merged.csv and run_info.json.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Destination directory for the random-word control concept_names_merged.csv and run_info.json.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for word sampling (default: 42).",
    )
    parser.add_argument(
        "--words-file",
        help="Optional newline-delimited word list file (one word/phrase per line).",
    )
    return parser.parse_args()


def load_words_file(path):
    words_path = Path(path)
    if not words_path.exists():
        raise FileNotFoundError(f"Words file not found: {words_path}")
    seen = set()
    words = []
    for line in words_path.read_text(encoding="utf-8").splitlines():
        word = line.strip()
        if word and word not in seen:
            seen.add(word)
            words.append(word)
    return words


def build_random_words(n, seed, words_file=None):
    """Return n unique random everyday words/phrases unrelated to medical concepts."""
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}.")
    if n == 0:
        return []

    rng = random.Random(seed)
    if words_file is not None:
        pool = load_words_file(words_file)
        if len(pool) < n:
            raise ValueError(
                f"Words file '{words_file}' has only {len(pool)} unique words, but {n} concepts are required."
            )
        return rng.sample(pool, n)

    nouns = list(dict.fromkeys(BUILTIN_NOUNS))
    rng.shuffle(nouns)
    if n <= len(nouns):
        return nouns[:n]

    modifiers = list(dict.fromkeys(BUILTIN_MODIFIERS))
    combos = [f"{mod} {noun}" for mod, noun in itertools.product(modifiers, nouns)]
    if len(nouns) + len(combos) < n:
        raise ValueError(
            f"Built-in pool supports at most {len(nouns) + len(combos)} unique concepts, "
            f"but {n} were requested. Pass --words-file with more words."
        )
    rng.shuffle(combos)
    return nouns + combos[: n - len(nouns)]


def read_merged_concepts(csv_path):
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found.")
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        required = {"unit", "concept_name", "merged_from"}
        if not required.issubset(fieldnames):
            raise ValueError(f"{csv_path} must contain columns: unit, concept_name, merged_from.")
        rows = []
        for row in reader:
            if not (row["concept_name"] or "").strip():
                continue
            rows.append(
                {
                    "unit": row["unit"].strip(),
                    "concept_name": row["concept_name"].strip(),
                    "merged_from": row["merged_from"].strip(),
                }
            )
    if not rows:
        raise ValueError(f"{csv_path} contains no valid concepts.")
    return rows


def main():
    args = parse_args()
    src_dir = Path(args.concept_dir)
    dst_dir = Path(args.output_dir)

    if src_dir.resolve() == dst_dir.resolve():
        raise ValueError("--output-dir must be different from --concept-dir to protect source concepts.")

    src_csv = src_dir / "concept_names_merged.csv"
    src_run_info_path = src_dir / "run_info.json"
    if not src_run_info_path.exists():
        raise FileNotFoundError(f"{src_run_info_path} not found.")

    rows = read_merged_concepts(src_csv)
    run_info = json.loads(src_run_info_path.read_text(encoding="utf-8"))

    random_words = build_random_words(len(rows), seed=args.seed, words_file=args.words_file)
    control_rows = [
        {
            "unit": row["unit"],
            "concept_name": word,
            "merged_from": row["merged_from"],
        }
        for row, word in zip(rows, random_words)
    ]

    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_csv = dst_dir / "concept_names_merged.csv"
    with dst_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["unit", "concept_name", "merged_from"])
        writer.writeheader()
        writer.writerows(control_rows)

    run_info["merged_concepts"] = str(dst_csv)
    run_info["random_word_control"] = True
    run_info["source_concept_dir"] = str(src_dir)
    run_info["random_seed"] = args.seed
    if args.words_file is not None:
        run_info["words_file"] = str(args.words_file)

    dst_run_info_path = dst_dir / "run_info.json"
    dst_run_info_path.write_text(json.dumps(run_info, indent=2), encoding="utf-8")

    print(f"Created {len(control_rows)} random-word control concepts in: {dst_dir}")
    print(f"- {dst_csv}")
    print(f"- {dst_run_info_path}")
    print("\nNext step: run annotation with the same config, split, and LLM settings as the default run:")
    print(
        f"python -m mcbm.scripts.annotate_concepts "
        f"--config configs/mydata.json "
        f"--concept-dir {dst_dir} "
        f"--split test "
        f"--llm-provider <provider> "
        f"--llm-model <model>"
    )


if __name__ == "__main__":
    main()
