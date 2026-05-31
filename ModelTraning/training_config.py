from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


MODEL_TRAINING_DIR = Path(__file__).resolve().parent

DEFAULTS: dict[str, Any] = {
    "test_fraction": 0.2,
    "seed": 1337,
    "sample_rate": 16000,
    "min_total_length": 32000,
    "clip_buffer_samples": 12000,
    "augmentation_rounds": 1,
    "augmentation_batch_size": 16,
    "background_dirs": [],
    "rir_dirs": [],
    "generic_negative_features": str(
        MODEL_TRAINING_DIR / "assets" / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
    ),
    "false_positive_validation_features": str(MODEL_TRAINING_DIR / "assets" / "validation_set_features.npy"),
    "negative_source_dir": None,
    "negative_test_samples": 2000,
    "steps": 10000,
    "batch_n_per_class": {
        "generic_negative": 1024,
        "positive": 50,
    },
    "model_type": "dnn",
    "layer_size": 32,
    "max_negative_weight": 1500,
    "target_false_positives_per_hour": 0.2,
    "convert_to_tflite": False,
}


def slugify(value: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "_" for char in value.strip())
    slug = "_".join(part for part in slug.split("_") if part)
    if not slug:
        raise ValueError("Nome da wake word/modelo nao pode ser vazio.")
    return slug


def infer_model_name(data_dir: Path) -> str:
    name = data_dir.name.strip()
    lowered = name.lower()
    for prefix in ("dataset-", "dataset_", "wakeword-", "wakeword_", "wake-word-", "wake-word_"):
        if lowered.startswith(prefix):
            name = name[len(prefix) :]
            break
    return slugify(name)


def load_config(path: Path | None) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULTS)
    if path is None:
        return config

    with path.open("r", encoding="utf-8") as f:
        config.update(json.load(f))

    base_dir = path.parent.resolve()
    for key in (
        "work_dir",
        "output_dir",
        "prepared_positive_dir",
        "generic_negative_features",
        "false_positive_validation_features",
        "negative_source_dir",
    ):
        if config.get(key):
            config[key] = str((base_dir / config[key]).resolve())

    config["background_dirs"] = [str((base_dir / item).resolve()) for item in config.get("background_dirs", [])]
    config["rir_dirs"] = [str((base_dir / item).resolve()) for item in config.get("rir_dirs", [])]
    return config


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(Path(args.config).resolve() if args.config else None)

    data_dir = Path(args.data_dir).resolve()
    if args.model_name:
        model_name = slugify(args.model_name)
    elif config.get("model_name"):
        model_name = slugify(str(config["model_name"]))
    else:
        model_name = infer_model_name(data_dir)

    config["model_name"] = model_name
    config["positive_source_dir"] = str(data_dir)
    config["work_dir"] = str(
        Path(args.work_dir).resolve()
        if args.work_dir
        else Path(config["work_dir"]).resolve()
        if config.get("work_dir")
        else MODEL_TRAINING_DIR / "work" / model_name
    )
    config["prepared_positive_dir"] = str(
        Path(args.prepared_positive_dir).resolve()
        if args.prepared_positive_dir
        else Path(config["prepared_positive_dir"]).resolve()
        if config.get("prepared_positive_dir")
        else Path(config["work_dir"]) / "positive_16k"
    )
    config["output_dir"] = str(
        Path(args.output_dir).resolve()
        if args.output_dir
        else Path(config["output_dir"]).resolve()
        if config.get("output_dir")
        else MODEL_TRAINING_DIR / "models"
    )

    if args.steps is not None:
        config["steps"] = args.steps
    if args.negative_source_dir:
        config["negative_source_dir"] = str(Path(args.negative_source_dir).resolve())
    if args.background_dir:
        config["background_dirs"] = [str(Path(item).resolve()) for item in args.background_dir]
    if args.rir_dir:
        config["rir_dirs"] = [str(Path(item).resolve()) for item in args.rir_dir]
    if args.convert_to_tflite:
        config["convert_to_tflite"] = True

    return config
