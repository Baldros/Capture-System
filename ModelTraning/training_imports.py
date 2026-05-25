from __future__ import annotations

import logging
from typing import Any


_TORCH_DEVICE_LOGGED = False


def require_training_imports() -> dict[str, Any]:
    global _TORCH_DEVICE_LOGGED
    try:
        import torch
        import onnxruntime
        from openwakeword.train import (
            Model as TrainModel,
            augment_clips,
            compute_features_from_generator,
            convert_onnx_to_tflite,
            mmap_batch_generator,
        )
        from openwakeword.utils import AudioFeatures
    except ImportError as exc:
        if isinstance(exc, ModuleNotFoundError) and exc.name:
            missing = f" Dependencia faltando: {exc.name}."
        else:
            missing = f" Erro de importacao: {exc}."
        raise SystemExit(
            "Dependencias de treino ausentes."
            f"{missing} Rode no venv de treino: pip install -r ModelTraning/requirements.txt"
        ) from exc

    imports = {
        "torch": torch,
        "onnxruntime": onnxruntime,
        "AudioFeatures": AudioFeatures,
        "augment_clips": augment_clips,
        "mmap_batch_generator": mmap_batch_generator,
        "TrainModel": TrainModel,
        "convert_onnx_to_tflite": convert_onnx_to_tflite,
        "compute_features_from_generator": compute_features_from_generator,
    }

    if not _TORCH_DEVICE_LOGGED:
        if torch.cuda.is_available():
            logging.info("PyTorch CUDA disponivel: %s", torch.version.cuda)
        else:
            logging.info("PyTorch CUDA indisponivel; treino rodara em CPU.")
        _TORCH_DEVICE_LOGGED = True

    return imports
