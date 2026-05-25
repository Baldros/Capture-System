from __future__ import annotations

import logging
import gc
import os
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm

try:
    from .dataset_split import wav_paths
    from .training_imports import require_training_imports
    from .training_progress import TrainingProgress, file_size
except ImportError:
    from dataset_split import wav_paths
    from training_imports import require_training_imports
    from training_progress import TrainingProgress, file_size


def asset_paths(directories: list[str]) -> list[str]:
    paths: list[str] = []
    for directory in directories:
        root = Path(directory)
        if root.exists():
            paths.extend(str(path) for path in root.rglob("*.wav"))
    return paths


def infer_total_length(
    positive_test_dir: Path,
    min_total_length: int,
    clip_buffer_samples: int,
    sample_rate: int = 16000,
) -> int:
    try:
        import scipy.io.wavfile
    except ImportError as exc:
        raise SystemExit("scipy e necessario para inferir a duracao dos clips.") from exc

    clips = wav_paths(positive_test_dir)
    if not clips:
        raise FileNotFoundError(f"Nenhum .wav encontrado em {positive_test_dir}")

    sample = clips[: min(50, len(clips))]
    lengths = []
    for clip in sample:
        sr, data = scipy.io.wavfile.read(clip)
        if sr != sample_rate:
            raise ValueError(f"{clip} esta em {sr} Hz; openWakeWord espera {sample_rate} Hz.")
        lengths.append(len(data))

    total_length = int(round(float(np.median(lengths)) / 1000) * 1000) + clip_buffer_samples
    if total_length < min_total_length or abs(total_length - min_total_length) <= 4000:
        return min_total_length
    return total_length


def onnx_feature_device(imports: dict[str, Any], progress: TrainingProgress) -> str:
    providers = set(imports["onnxruntime"].get_available_providers())
    if "CUDAExecutionProvider" in providers:
        return "gpu"

    torch = imports["torch"]
    if torch.cuda.is_available():
        progress.info("PyTorch CUDA disponivel, mas onnxruntime-gpu ausente; features serao extraidas em CPU.")
    return "cpu"


def compute_features_from_generator_safe(
    generator: Any,
    n_total: int,
    clip_duration: int,
    output_file: Path,
    device: str,
    ncpu: int,
    imports: dict[str, Any],
) -> None:
    audio_features = imports["AudioFeatures"](device=device)
    n_feature_cols = audio_features.get_embedding_shape(clip_duration / 16000)
    output_shape = (n_total, n_feature_cols[0], n_feature_cols[1])
    fp = None

    try:
        fp = open_memmap(str(output_file), mode="w+", dtype=np.float32, shape=output_shape)
        row_counter = 0

        try:
            audio_data = next(generator)
        except StopIteration as exc:
            raise ValueError("O gerador de audio nao produziu clips para extrair features.") from exc

        batch_size = audio_data.shape[0]
        if batch_size > n_total:
            raise ValueError(
                f"O valor de n_total ({n_total}) e menor que o batch ({batch_size}). "
                "Aumente n_total para ser >= batch size."
            )

        features = audio_features.embed_clips(audio_data, batch_size=batch_size)
        fp[row_counter : row_counter + features.shape[0], :, :] = features
        row_counter += features.shape[0]
        fp.flush()

        for audio_data in tqdm(generator, total=n_total // batch_size, desc="Computing features"):
            if row_counter >= n_total:
                break

            features = audio_features.embed_clips(audio_data, batch_size=batch_size, ncpu=ncpu)
            if row_counter + features.shape[0] > n_total:
                features = features[: n_total - row_counter]

            fp[row_counter : row_counter + features.shape[0], :, :] = features
            row_counter += features.shape[0]
            fp.flush()

        if row_counter < n_total:
            temp_file = output_file.with_name(f"{output_file.stem}.{uuid.uuid4().hex}.tmp.npy")
            trimmed = open_memmap(
                str(temp_file),
                mode="w+",
                dtype=np.float32,
                shape=(row_counter, n_feature_cols[0], n_feature_cols[1]),
            )
            try:
                for start in tqdm(range(0, row_counter, 1024), desc="Trimming empty rows"):
                    end = min(start + 1024, row_counter)
                    trimmed[start:end] = fp[start:end].copy()
                    trimmed.flush()
            finally:
                trimmed.flush()
                del trimmed

            fp.flush()
            del fp
            fp = None
            gc.collect()
            os.replace(temp_file, output_file)
    finally:
        if fp is not None:
            with suppress(Exception):
                fp.flush()
            del fp
            gc.collect()


def compute_feature_file(
    clip_paths: list[str],
    output_file: Path,
    total_length: int,
    config: dict[str, Any],
    imports: dict[str, Any],
    overwrite: bool,
    progress: TrainingProgress | None = None,
) -> None:
    progress = progress or TrainingProgress(enabled=False)
    if output_file.exists() and not overwrite:
        logging.info("Skipping existing features: %s", output_file)
        progress.skip(f"Features existentes: {output_file} ({file_size(output_file)})")
        return
    if not clip_paths:
        raise FileNotFoundError(f"Nenhum clip encontrado para gerar {output_file}")

    n_total = len(clip_paths) * int(config["augmentation_rounds"])
    background_paths = asset_paths(config.get("background_dirs", []))
    rir_paths = asset_paths(config.get("rir_dirs", []))
    device = onnx_feature_device(imports, progress)
    progress.summary(
        "Geracao de feature file",
        {
            "saida": output_file,
            "clips base": len(clip_paths),
            "clips com augmentation": n_total,
            "total_length": total_length,
            "device": device,
            "background clips": len(background_paths),
            "rir clips": len(rir_paths),
        },
    )
    generator = imports["augment_clips"](
        clip_paths * int(config["augmentation_rounds"]),
        total_length=total_length,
        batch_size=int(config["augmentation_batch_size"]),
        background_clip_paths=background_paths,
        RIR_paths=rir_paths,
    )

    n_cpus = max(1, (os.cpu_count() or 2) // 2)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with progress.timed(f"Computando features em {output_file.name}", heartbeat_seconds=20):
        compute_features_from_generator_safe(
            generator,
            n_total=n_total,
            clip_duration=total_length,
            output_file=output_file,
            device=device,
            ncpu=n_cpus if device == "cpu" else 1,
            imports=imports,
        )
    progress.ok(f"Arquivo salvo: {output_file} ({file_size(output_file)})")


def make_negative_test_features(
    generic_features_path: Path,
    output_file: Path,
    input_shape: tuple[int, int],
    n_samples: int,
    overwrite: bool,
    progress: TrainingProgress | None = None,
) -> None:
    progress = progress or TrainingProgress(enabled=False)
    if output_file.exists() and not overwrite:
        logging.info("Skipping existing negative test features: %s", output_file)
        progress.skip(f"Features negativas de teste existentes: {output_file} ({file_size(output_file)})")
        return

    progress.summary(
        "Features negativas de teste",
        {
            "origem": generic_features_path,
            "saida": output_file,
            "amostras": n_samples,
            "input_shape": input_shape,
        },
    )
    source = np.load(generic_features_path, mmap_mode="r")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if source.ndim == 3 and tuple(source.shape[1:]) == input_shape:
        np.save(output_file, np.asarray(source[:n_samples], dtype=np.float32))
        progress.ok(f"Arquivo salvo: {output_file} ({file_size(output_file)})")
        return

    if source.ndim == 3 and source.shape[2] == input_shape[1]:
        source_frames = int(source.shape[1])
        target_frames = int(input_shape[0])
        rows_needed = int(np.ceil(n_samples * target_frames / source_frames)) + 1
        flat = np.asarray(source[:rows_needed], dtype=np.float32).reshape(-1, input_shape[1])
        usable_samples = min(n_samples, flat.shape[0] // target_frames)
        windows = flat[: usable_samples * target_frames].reshape(usable_samples, target_frames, input_shape[1])
        np.save(output_file, windows)
        progress.ok(f"Arquivo salvo: {output_file} ({file_size(output_file)})")
        return

    if source.ndim == 2 and source.shape[1] == input_shape[1]:
        n = min(n_samples, max(1, source.shape[0] - input_shape[0]))
        windows = np.asarray([source[i : i + input_shape[0]] for i in range(n)], dtype=np.float32)
        np.save(output_file, windows)
        progress.ok(f"Arquivo salvo: {output_file} ({file_size(output_file)})")
        return

    raise ValueError(
        f"Formato inesperado em {generic_features_path}: {source.shape}; "
        f"esperado (*, {input_shape[0]}, {input_shape[1]}) ou (*, {input_shape[1]})."
    )


def prepare_features(
    config: dict[str, Any],
    overwrite: bool = False,
    progress: TrainingProgress | None = None,
) -> tuple[Path, tuple[int, int]]:
    progress = progress or TrainingProgress(enabled=False)
    imports = require_training_imports()
    work_dir = Path(config["work_dir"])
    feature_dir = work_dir / "features"
    positive_train_dir = work_dir / "positive_train"
    positive_test_dir = work_dir / "positive_test"

    progress.info(f"Analisando duracao dos WAVs em {positive_test_dir}")
    total_length = infer_total_length(
        positive_test_dir,
        min_total_length=int(config["min_total_length"]),
        clip_buffer_samples=int(config["clip_buffer_samples"]),
        sample_rate=int(config["sample_rate"]),
    )
    logging.info("Using total_length=%s samples", total_length)
    progress.ok(f"total_length definido em {total_length} samples")

    compute_feature_file(
        wav_paths(positive_train_dir),
        feature_dir / "positive_features_train.npy",
        total_length,
        config,
        imports,
        overwrite,
        progress=progress,
    )
    compute_feature_file(
        wav_paths(positive_test_dir),
        feature_dir / "positive_features_test.npy",
        total_length,
        config,
        imports,
        overwrite,
        progress=progress,
    )

    positive_test = np.load(feature_dir / "positive_features_test.npy", mmap_mode="r")
    input_shape = tuple(int(i) for i in positive_test.shape[1:])
    progress.ok(f"input_shape detectado: {input_shape}")

    negative_source = config.get("negative_source_dir")
    if negative_source:
        negative_dir = Path(negative_source)
        progress.info(f"Usando negativos locais em {negative_dir}")
        compute_feature_file(
            wav_paths(negative_dir / "train"),
            feature_dir / "negative_features_train.npy",
            total_length,
            config,
            imports,
            overwrite,
            progress=progress,
        )
        compute_feature_file(
            wav_paths(negative_dir / "test"),
            feature_dir / "negative_features_test.npy",
            total_length,
            config,
            imports,
            overwrite,
            progress=progress,
        )
    else:
        make_negative_test_features(
            Path(config["generic_negative_features"]),
            feature_dir / "negative_features_test.npy",
            input_shape,
            int(config["negative_test_samples"]),
            overwrite,
            progress=progress,
        )

    return feature_dir, input_shape
