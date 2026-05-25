from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    from .audio_preparation import prepare_training_audio
    from .dataset_split import prepare_positive_split
    from .feature_preparation import prepare_features
    from .model_training import train_model
    from .training_progress import TrainingProgress
except ImportError:
    from audio_preparation import prepare_training_audio
    from dataset_split import prepare_positive_split
    from feature_preparation import prepare_features
    from model_training import train_model
    from training_progress import TrainingProgress


def run_training_pipeline(
    config: dict[str, Any],
    stage: str,
    overwrite_audio: bool = False,
    overwrite_split: bool = False,
    overwrite_features: bool = False,
    progress: TrainingProgress | None = None,
) -> Path | None:
    progress = progress or TrainingProgress()
    planned_stages = []
    if stage in {"prepare", "features", "all"}:
        planned_stages.extend(["audio", "split"])
    if stage in {"features", "all"}:
        planned_stages.append("features")
    if stage in {"train", "all"}:
        planned_stages.append("train")
    total_stages = len(planned_stages)
    stage_number = 1

    if stage in {"prepare", "features", "all"}:
        source_dir = Path(config["positive_source_dir"])
        work_dir = Path(config["work_dir"])
        prepared_positive_dir = Path(config["prepared_positive_dir"])
        progress.stage(
            stage_number,
            total_stages,
            "Preparacao dos audios positivos",
            [
                f"origem: {source_dir}",
                f"saida: {prepared_positive_dir}",
                f"formato: {config['sample_rate']} Hz, mono, PCM_16",
            ],
        )
        with progress.timed("Convertendo audios para o formato do openWakeWord", heartbeat_seconds=15):
            audio_manifest = prepare_training_audio(
                source_dir=source_dir,
                output_dir=prepared_positive_dir,
                sample_rate=int(config["sample_rate"]),
                overwrite=overwrite_audio,
                show_progress=progress.enabled,
            )
        progress.summary(
            "Audios preparados",
            {
                "arquivos": audio_manifest.get("source_wav_count", 0),
                "sample_rate": audio_manifest.get("sample_rate", config["sample_rate"]),
                "audio_rebuilt": audio_manifest.get("audio_rebuilt", False),
            },
        )
        audio_rebuilt = bool(audio_manifest.get("audio_rebuilt"))
        stage_number += 1

        progress.stage(
            stage_number,
            total_stages,
            "Split dos audios positivos",
            [
                f"origem: {prepared_positive_dir}",
                f"work-dir: {work_dir}",
                f"test_fraction: {config['test_fraction']}",
            ],
        )
        with progress.timed("Preparando split positivo", heartbeat_seconds=15):
            manifest = prepare_positive_split(
                source_dir=prepared_positive_dir,
                work_dir=work_dir,
                test_fraction=float(config["test_fraction"]),
                seed=int(config["seed"]),
                overwrite=overwrite_split or audio_rebuilt,
                show_progress=progress.enabled,
            )
        progress.summary("Arquivos do split", manifest.get("counts", {}))
        split_rebuilt = bool(manifest.get("split_rebuilt"))
        stage_number += 1
    else:
        split_rebuilt = False

    if stage in {"features", "all"}:
        feature_overwrite = overwrite_features or split_rebuilt
        if split_rebuilt and not overwrite_features:
            progress.info("Split refeito; features existentes serao regeneradas para acompanhar os WAVs atuais.")
        progress.stage(
            stage_number,
            total_stages,
            "Extracao de features",
            [
                f"features: {Path(config['work_dir']) / 'features'}",
                f"augmentation_rounds: {config['augmentation_rounds']}",
                f"batch_size: {config['augmentation_batch_size']}",
            ],
        )
        feature_dir, input_shape = prepare_features(config, overwrite=feature_overwrite, progress=progress)
        stage_number += 1
    elif stage == "train":
        feature_dir = Path(config["work_dir"]) / "features"
        positive_test_features = feature_dir / "positive_features_test.npy"
        if not positive_test_features.exists():
            raise SystemExit(
                f"Features ausentes: {positive_test_features}. "
                "Rode primeiro com --stage features ou use --stage all."
            )
        progress.info(f"Carregando input_shape existente de {positive_test_features}")
        input_shape = tuple(np.load(positive_test_features, mmap_mode="r").shape[1:])
    else:
        return None

    if stage in {"train", "all"}:
        progress.stage(
            stage_number,
            total_stages,
            "Treinamento e exportacao",
            [
                f"steps base: {config['steps']}",
                f"modelo: {config['model_name']}",
                f"saida: {Path(config['output_dir'])}",
            ],
        )
        return train_model(config, feature_dir, input_shape, progress=progress)

    return None
