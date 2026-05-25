from __future__ import annotations

import json
import logging
import math
import shutil
from pathlib import Path
from typing import Any

try:
    from .training_progress import progress_iter
except ImportError:
    from training_progress import progress_iter


def wav_paths(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.wav"))


def source_fingerprint(paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "name": path.name,
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in paths
    ]


def wav_matches_training_format(path: Path, sample_rate: int) -> bool:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise SystemExit("soundfile e necessario para preparar WAVs de treino.") from exc

    info = sf.info(str(path))
    return info.samplerate == sample_rate and info.channels == 1 and info.subtype == "PCM_16"


def write_training_wav(source: Path, destination: Path, sample_rate: int) -> None:
    try:
        import numpy as np
        import soundfile as sf
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise SystemExit("soundfile, numpy e scipy sao necessarios para preparar WAVs de treino.") from exc

    audio, source_sample_rate = sf.read(str(source), always_2d=True, dtype="float32")
    mono = audio.mean(axis=1)
    if source_sample_rate != sample_rate:
        divisor = math.gcd(source_sample_rate, sample_rate)
        mono = resample_poly(mono, sample_rate // divisor, source_sample_rate // divisor).astype("float32")

    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(destination), np.clip(mono, -1.0, 1.0), sample_rate, subtype="PCM_16")


def read_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def output_is_current(source_dir: Path, output_dir: Path, sample_rate: int, manifest_path: Path) -> bool:
    source_files = wav_paths(source_dir)
    output_files = wav_paths(output_dir)
    if not source_files or len(source_files) != len(output_files):
        return False

    source_names = {path.name for path in source_files}
    output_names = {path.name for path in output_files}
    if source_names != output_names:
        return False

    manifest = read_manifest(manifest_path)
    if manifest is None or manifest.get("sample_rate") != sample_rate:
        return False
    if manifest.get("source_wav_count") != len(source_files):
        return False
    if manifest.get("source_files") != source_fingerprint(source_files):
        return False

    return all(wav_matches_training_format(path, sample_rate) for path in output_files)


def prepare_training_audio(
    source_dir: Path,
    output_dir: Path,
    sample_rate: int,
    overwrite: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    source_files = wav_paths(source_dir)
    if not source_files:
        raise FileNotFoundError(f"Nenhum .wav encontrado em {source_dir}")

    manifest_path = output_dir / "audio_manifest.json"
    if output_dir.exists() and not overwrite and output_is_current(source_dir, output_dir, sample_rate, manifest_path):
        logging.info("Usando audios preparados existentes em %s", output_dir)
        manifest = read_manifest(manifest_path) or {}
        manifest["audio_rebuilt"] = False
        return manifest

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for source in progress_iter(
        source_files,
        total=len(source_files),
        desc="Convertendo WAVs",
        unit="wav",
        enabled=show_progress,
    ):
        write_training_wav(source, output_dir / source.name, sample_rate)

    metadata_path = source_dir / "metadata.json"
    if metadata_path.exists():
        shutil.copy2(metadata_path, output_dir / "metadata.json")

    manifest = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "source_wav_count": len(source_files),
        "source_files": source_fingerprint(source_files),
        "sample_rate": sample_rate,
        "channels": 1,
        "subtype": "PCM_16",
        "audio_rebuilt": True,
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest
