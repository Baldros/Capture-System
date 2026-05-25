from __future__ import annotations

import argparse
import logging
import sys

try:
    from .training_config import build_config
    from .training_pipeline import run_training_pipeline
    from .training_progress import TrainingProgress
except ImportError:
    from training_config import build_config
    from training_pipeline import run_training_pipeline
    from training_progress import TrainingProgress


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Treina um modelo openWakeWord a partir de uma pasta de WAVs positivos."
    )
    parser.add_argument("data_dir", help="Pasta com os .wav positivos da wake word.")
    parser.add_argument("--model-name", help="Nome do arquivo .onnx de saida. Default: inferido pelo nome da pasta.")
    parser.add_argument("--config", default=None, help="JSON opcional com parametros avancados de treino.")
    parser.add_argument("--work-dir", help="Diretorio de trabalho para splits/features.")
    parser.add_argument("--prepared-positive-dir", help="Diretorio derivado com WAVs positivos em 16 kHz mono.")
    parser.add_argument("--output-dir", help="Diretorio onde o .onnx sera salvo.")
    parser.add_argument("--negative-source-dir", help="Pasta opcional com negativos locais em subpastas train/ e test/.")
    parser.add_argument("--background-dir", action="append", help="Pasta de ruidos/fundos para augmentation.")
    parser.add_argument("--rir-dir", action="append", help="Pasta de room impulse responses para augmentation.")
    parser.add_argument("--steps", type=int, help="Sobrescreve a quantidade de steps do treino.")
    parser.add_argument("--stage", choices=["prepare", "features", "train", "all"], default="all")
    parser.add_argument("--overwrite-audio", action="store_true")
    parser.add_argument("--overwrite-split", action="store_true")
    parser.add_argument("--overwrite-features", action="store_true")
    parser.add_argument("--convert-to-tflite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--no-visual-progress", action="store_true", help="Desativa os indicadores visuais extras.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(levelname)s] %(message)s",
        stream=sys.stdout,
        force=True,
    )
    config = build_config(args)
    progress = TrainingProgress(enabled=not args.no_visual_progress)
    progress.banner(
        "Pipeline de treino openWakeWord",
        [
            f"data_dir: {config['positive_source_dir']}",
            f"work_dir: {config['work_dir']}",
            f"output_dir: {config['output_dir']}",
        ],
    )
    progress.summary(
        "Resumo",
        {
            "stage": args.stage,
            "model_name": config["model_name"],
            "steps": config["steps"],
            "overwrite_audio": args.overwrite_audio,
            "overwrite_split": args.overwrite_split,
            "overwrite_features": args.overwrite_features,
            "convert_to_tflite": config.get("convert_to_tflite", False),
        },
    )

    onnx_path = run_training_pipeline(
        config,
        stage=args.stage,
        overwrite_audio=args.overwrite_audio,
        overwrite_split=args.overwrite_split,
        overwrite_features=args.overwrite_features,
        progress=progress,
    )
    if onnx_path is not None:
        progress.ok(f"Modelo salvo em: {onnx_path}")

    return 0
