from __future__ import annotations

import json
import logging
import random
import shutil
from pathlib import Path
from typing import Any

try:
    from .training_progress import progress_iter
except ImportError:
    from training_progress import progress_iter


def wav_paths(directory: Path) -> list[str]:
    return [str(path) for path in sorted(directory.glob("*.wav"))]


def split_counts(train_dir: Path, test_dir: Path) -> dict[str, int]:
    return {
        "positive_train": len(wav_paths(train_dir)) if train_dir.exists() else 0,
        "positive_test": len(wav_paths(test_dir)) if test_dir.exists() else 0,
    }


def prepare_positive_split(
    source_dir: Path,
    work_dir: Path,
    test_fraction: float,
    seed: int,
    overwrite: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    wavs = sorted(source_dir.glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"Nenhum .wav encontrado em {source_dir}")

    source_names = {path.name for path in wavs}
    train_dir = work_dir / "positive_train"
    test_dir = work_dir / "positive_test"
    manifest_path = work_dir / "split_manifest.json"
    if train_dir.exists() and test_dir.exists() and not overwrite:
        counts = split_counts(train_dir, test_dir)
        split_names = {Path(path).name for path in wav_paths(train_dir) + wav_paths(test_dir)}
        if split_names == source_names:
            logging.info("Usando split existente em %s", work_dir)
            if manifest_path.exists():
                with manifest_path.open("r", encoding="utf-8") as f:
                    manifest = json.load(f)
                manifest["counts"] = counts
                manifest["source_wav_count"] = len(wavs)
                manifest["split_rebuilt"] = False
                return manifest
            return {
                "source_dir": str(source_dir),
                "source_wav_count": len(wavs),
                "test_fraction": test_fraction,
                "seed": seed,
                "split_rebuilt": False,
                "counts": counts,
            }

        logging.warning(
            "Split existente em %s esta desatualizado: origem tem %s WAVs, split tem %s. Recriando split.",
            work_dir,
            len(wavs),
            len(split_names),
        )
        for directory in (train_dir, test_dir):
            if directory.exists():
                shutil.rmtree(directory)
        if manifest_path.exists():
            manifest_path.unlink()
    elif (train_dir.exists() or test_dir.exists()) and not overwrite:
        logging.warning("Split parcial encontrado em %s. Recriando split.", work_dir)
        for directory in (train_dir, test_dir):
            if directory.exists():
                shutil.rmtree(directory)
        if manifest_path.exists():
            manifest_path.unlink()

    if overwrite:
        for directory in (train_dir, test_dir):
            if directory.exists():
                shutil.rmtree(directory)
        if manifest_path.exists():
            manifest_path.unlink()

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    if len(wavs) < 2:
        raise ValueError(
            f"Sao necessarios ao menos 2 WAVs positivos para o split treino/teste; "
            f"encontrados {len(wavs)} em {source_dir}."
        )

    shuffled = wavs[:]
    random.Random(seed).shuffle(shuffled)
    # Garante pelo menos 1 clip em cada lado do split.
    n_test = min(max(1, int(len(shuffled) * test_fraction)), len(shuffled) - 1)
    splits = {
        test_dir: shuffled[:n_test],
        train_dir: shuffled[n_test:],
    }

    for split_dir, files in splits.items():
        for source in progress_iter(
            files,
            total=len(files),
            desc=f"Copiando {split_dir.name}",
            unit="wav",
            enabled=show_progress,
        ):
            shutil.copy2(source, split_dir / source.name)

    metadata_path = source_dir / "metadata.json"
    if metadata_path.exists():
        shutil.copy2(metadata_path, work_dir / "source_metadata.json")

    manifest = {
        "source_dir": str(source_dir),
        "source_wav_count": len(wavs),
        "test_fraction": test_fraction,
        "seed": seed,
        "split_rebuilt": True,
        "counts": {
            "positive_train": len(splits[train_dir]),
            "positive_test": len(splits[test_dir]),
        },
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest
