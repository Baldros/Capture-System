from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .training_imports import require_training_imports
    from .training_progress import TrainingProgress, file_size
except ImportError:
    from training_imports import require_training_imports
    from training_progress import TrainingProgress, file_size


def transform_to_shape(x: np.ndarray, n_frames: int) -> np.ndarray:
    if n_frames == x.shape[1]:
        return x
    x = np.vstack(x)
    return np.array([x[i : i + n_frames, :] for i in range(0, x.shape[0] - n_frames, n_frames)])


def npy_shape(path: Path) -> tuple[int, ...]:
    return tuple(int(i) for i in np.load(path, mmap_mode="r").shape)


def train_model(
    config: dict[str, Any],
    feature_dir: Path,
    input_shape: tuple[int, int],
    progress: TrainingProgress | None = None,
) -> Path:
    progress = progress or TrainingProgress(enabled=False)
    imports = require_training_imports()
    torch = imports["torch"]
    mmap_batch_generator = imports["mmap_batch_generator"]
    TrainModel = imports["TrainModel"]

    generic_negative_features = Path(config["generic_negative_features"])
    false_positive_features = Path(config["false_positive_validation_features"])
    if not generic_negative_features.exists():
        raise FileNotFoundError(f"Feature negativa generica ausente: {generic_negative_features}")
    if not false_positive_features.exists():
        raise FileNotFoundError(f"Feature de validacao FP ausente: {false_positive_features}")

    feature_data_files = {
        "generic_negative": str(generic_negative_features),
        "positive": str(feature_dir / "positive_features_train.npy"),
    }
    if (feature_dir / "negative_features_train.npy").exists():
        feature_data_files["local_negative"] = str(feature_dir / "negative_features_train.npy")

    progress.summary(
        "Arquivos de treino",
        {
            "positive train": f"{feature_dir / 'positive_features_train.npy'} ({file_size(feature_dir / 'positive_features_train.npy')})",
            "positive test": f"{feature_dir / 'positive_features_test.npy'} ({file_size(feature_dir / 'positive_features_test.npy')})",
            "negative test": f"{feature_dir / 'negative_features_test.npy'} ({file_size(feature_dir / 'negative_features_test.npy')})",
            "generic negative": f"{generic_negative_features} ({file_size(generic_negative_features)})",
            "false positive val": f"{false_positive_features} ({file_size(false_positive_features)})",
        },
    )

    data_transforms = {
        key: (lambda x, n=input_shape[0]: transform_to_shape(x, n))
        for key in feature_data_files
        if key != "positive"
    }
    label_transforms = {
        key: (lambda x, label=(1 if key == "positive" else 0): [label for _ in x])
        for key in feature_data_files
    }

    batch_n_per_class = copy.deepcopy(config["batch_n_per_class"])
    if "local_negative" in feature_data_files and "local_negative" not in batch_n_per_class:
        batch_n_per_class["local_negative"] = batch_n_per_class.get("generic_negative", 512)

    batch_generator = mmap_batch_generator(
        feature_data_files,
        n_per_class=batch_n_per_class,
        data_transform_funcs=data_transforms,
        label_transform_funcs=label_transforms,
    )

    class IterDataset(torch.utils.data.IterableDataset):
        def __init__(self, generator):
            self.generator = generator

        def __iter__(self):
            return self.generator

    n_workers = 0 if os.name == "nt" else max(1, (os.cpu_count() or 2) // 2)
    loader_kwargs = {
        "batch_size": None,
        "num_workers": n_workers,
    }
    if n_workers > 0:
        loader_kwargs["prefetch_factor"] = 16
    train_loader = torch.utils.data.DataLoader(IterDataset(batch_generator), **loader_kwargs)

    progress.summary(
        "Configuracao do treino",
        {
            "input_shape": input_shape,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "dataloader workers": n_workers,
            "steps base": int(config["steps"]),
            "steps aproximados": int(config["steps"]) + 2 * max(1, int(int(config["steps"]) / 10)),
            "batch_n_per_class": batch_n_per_class,
            "model_type": config["model_type"],
            "layer_size": int(config["layer_size"]),
        },
    )

    progress.info(f"Shape positive train: {npy_shape(feature_dir / 'positive_features_train.npy')}")
    progress.info(f"Shape positive test: {npy_shape(feature_dir / 'positive_features_test.npy')}")
    progress.info(f"Shape negative test: {npy_shape(feature_dir / 'negative_features_test.npy')}")

    with progress.timed("Carregando validacao de false positives", heartbeat_seconds=15):
        fp_features = np.load(false_positive_features)
        fp_windows = np.array(
            [fp_features[i : i + input_shape[0]] for i in range(0, fp_features.shape[0] - input_shape[0], 1)]
        )
        fp_labels = np.zeros(fp_windows.shape[0]).astype(np.float32)
    progress.ok(f"False-positive windows: {fp_windows.shape}")
    false_positive_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(fp_windows), torch.from_numpy(fp_labels)),
        batch_size=len(fp_labels),
    )

    with progress.timed("Montando validacao balanceada", heartbeat_seconds=15):
        pos_val = np.load(feature_dir / "positive_features_test.npy")
        neg_val = np.load(feature_dir / "negative_features_test.npy")
    labels = np.hstack((np.ones(pos_val.shape[0]), np.zeros(neg_val.shape[0]))).astype(np.float32)
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(np.vstack((pos_val, neg_val))), torch.from_numpy(labels)),
        batch_size=len(labels),
    )
    progress.ok(f"Validacao balanceada: positivos={pos_val.shape[0]}, negativos={neg_val.shape[0]}")

    model = TrainModel(
        n_classes=1,
        input_shape=input_shape,
        model_type=config["model_type"],
        layer_dim=int(config["layer_size"]),
        seconds_per_example=1280 * input_shape[0] / 16000,
    )
    progress.info("O openWakeWord exibira barras 'Training' para cada sequencia interna.")
    with progress.timed("Executando auto_train do openWakeWord", heartbeat_seconds=30):
        best_model = model.auto_train(
            X_train=train_loader,
            X_val=val_loader,
            false_positive_val_data=false_positive_loader,
            steps=int(config["steps"]),
            max_negative_weight=int(config["max_negative_weight"]),
            target_fp_per_hour=float(config["target_false_positives_per_hour"]),
        )

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with progress.timed("Exportando modelo ONNX", heartbeat_seconds=15):
        model.export_model(best_model, config["model_name"], str(output_dir))
    onnx_path = output_dir / f"{config['model_name']}.onnx"
    progress.ok(f"ONNX salvo: {onnx_path} ({file_size(onnx_path)})")

    if config.get("convert_to_tflite"):
        tflite_path = output_dir / f"{config['model_name']}.tflite"
        with progress.timed("Convertendo ONNX para TFLite", heartbeat_seconds=15):
            imports["convert_onnx_to_tflite"](str(onnx_path), str(tflite_path))
        progress.ok(f"TFLite salvo: {tflite_path} ({file_size(tflite_path)})")

    return onnx_path
