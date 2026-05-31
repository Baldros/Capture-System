from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path


ASSETS = {
    "openwakeword_features_ACAV100M_2000_hrs_16bit.npy": (
        "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/"
        "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
    ),
    "validation_set_features.npy": (
        "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/"
        "validation_set_features.npy"
    ),
}

DEFAULT_ASSETS_DIR = Path(__file__).with_name("assets")


def download_file(url: str, output_path: Path, overwrite: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0 and not overwrite:
        print(f"[skip] {output_path} already exists")
        return

    print(f"[download] {url}")
    # Baixa para um arquivo temporario e renomeia ao final, para que um download
    # interrompido nao deixe um arquivo truncado/0-byte que seria considerado
    # "existente" e nunca mais re-baixado.
    tmp_path = output_path.with_name(f"{output_path.name}.{os.getpid()}.part")
    try:
        with urllib.request.urlopen(url) as response, tmp_path.open("wb") as target:
            total = int(response.headers.get("Content-Length", "0") or 0)
            done = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
                done += len(chunk)
                if total:
                    pct = done / total * 100
                    print(f"\r  {pct:5.1f}% {done / (1024 ** 2):.1f} MB", end="")
            print()
        if done == 0:
            raise IOError(f"Download vazio de {url}")
        if total and done != total:
            raise IOError(f"Download incompleto de {url}: {done} de {total} bytes")
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def download_openwakeword_models(overwrite: bool = False) -> None:
    try:
        import openwakeword
    except ImportError as exc:
        raise SystemExit(
            "openwakeword nao esta instalado. Rode no venv de treino: "
            "pip install -r ModelTraning/requirements.txt"
        ) from exc

    package_dir = Path(openwakeword.__file__).resolve().parent
    models_dir = package_dir / "resources" / "models"
    urls: dict[str, str] = {}

    for metadata in openwakeword.FEATURE_MODELS.values():
        url = metadata["download_url"]
        urls[Path(url).name] = url
        if url.endswith(".tflite"):
            onnx_url = url.replace(".tflite", ".onnx")
            urls[Path(onnx_url).name] = onnx_url

    for metadata in openwakeword.VAD_MODELS.values():
        url = metadata["download_url"]
        urls[Path(url).name] = url

    print(f"[openwakeword] internal models -> {models_dir}")
    for filename, url in urls.items():
        download_file(url, models_dir / filename, overwrite=overwrite)


def download_mit_rirs(output_dir: Path, limit: int | None = None) -> None:
    try:
        import datasets
        import numpy as np
        import scipy.io.wavfile
        from tqdm import tqdm
    except ImportError as exc:
        raise SystemExit(
            "Para baixar RIRs instale as dependencias de treino: "
            "pip install -r ModelTraning/requirements.txt"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = datasets.load_dataset(
        "davidscripka/MIT_environmental_impulse_responses",
        split="train",
        streaming=True,
    )

    for idx, row in enumerate(tqdm(dataset, desc="MIT RIRs")):
        if limit is not None and idx >= limit:
            break
        name = Path(row["audio"]["path"]).name
        audio = (row["audio"]["array"] * 32767).astype(np.int16)
        scipy.io.wavfile.write(output_dir / name, 16000, audio)


def main() -> int:
    parser = argparse.ArgumentParser(description="Baixa assets usados pelo treino do openWakeWord.")
    parser.add_argument("--assets-dir", default=str(DEFAULT_ASSETS_DIR))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--with-rirs", action="store_true", help="Baixa RIRs do dataset MIT.")
    parser.add_argument("--rir-limit", type=int, default=None)
    args = parser.parse_args()

    assets_dir = Path(args.assets_dir).resolve()
    download_openwakeword_models(overwrite=args.overwrite)

    for filename, url in ASSETS.items():
        download_file(url, assets_dir / filename, overwrite=args.overwrite)

    if args.with_rirs:
        download_mit_rirs(assets_dir / "mit_rirs", limit=args.rir_limit)

    return 0


if __name__ == "__main__":
    sys.exit(main())
