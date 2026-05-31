import asyncio
import json
import math
import os
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from openwakeword.model import Model

# ─── configuração de captura ──────────────────────────────────────────────────

SAMPLE_RATE = 16_000   # Hz — padrão para voz
CHUNK_SIZE  = 1_280    # frames por callback (~80 ms)
CHANNELS    = 1
WAKEWORD_MODEL_PATH = Path(__file__).resolve().parent / "ModelTraning" / "models" / "atlas.onnx"
WAKEWORD_THRESHOLD = float(os.getenv("WAKEWORD_THRESHOLD", "0.5"))
GAIN        = 8.0      # multiplicador de ganho para visualização

app = FastAPI()

# estado compartilhado entre a thread de áudio e o event loop async
class AudioState:
    def __init__(self):
        self.capturing   = False
        self.clients:  list[WebSocket] = []
        self.lock      = threading.Lock()
        self.loop:     asyncio.AbstractEventLoop | None = None

state = AudioState()

# inicializa o modelo openwakeword com ONNX
WAKEWORD_MODELS = ["alexa"]
if WAKEWORD_MODEL_PATH.exists():
    WAKEWORD_MODELS.append(str(WAKEWORD_MODEL_PATH))

oww_model = Model(wakeword_models=WAKEWORD_MODELS, inference_framework="onnx")
WAKEWORD_NAMES = list(oww_model.models.keys())

# ─── cálculos do sinal ────────────────────────────────────────────────────────

def compute_rms(chunk: np.ndarray) -> float:
    """Energia RMS normalizada 0-1 com ganho aplicado."""
    rms = math.sqrt(np.mean(chunk.astype(np.float64) ** 2))
    return min((rms / 32768.0) * GAIN, 1.0)

def compute_waveform(chunk: np.ndarray, points: int = 80) -> list[float]:
    """Reduz o chunk para N pontos de amplitude normalizada."""
    chunk_f = chunk.astype(np.float32) / 32768.0 * GAIN
    # usa a média absoluta de cada segmento em vez de um ponto único
    seg = max(1, len(chunk_f) // points)
    samples = []
    for i in range(points):
        start = i * seg
        end   = min(start + seg, len(chunk_f))
        samples.append(float(min(np.mean(np.abs(chunk_f[start:end])), 1.0)))
    return samples

def predict_wakeword(chunk: np.ndarray) -> tuple[bool, str | None, float, dict[str, float], str | None]:
    try:
        prediction = oww_model.predict(chunk)
    except Exception as exc:
        return False, None, 0.0, {}, f"{type(exc).__name__}: {exc}"

    scores = {name: float(score) for name, score in prediction.items()}
    if not scores:
        return False, None, 0.0, scores, None

    wake_name, wake_score = max(scores.items(), key=lambda item: item[1])
    return wake_score > WAKEWORD_THRESHOLD, wake_name, wake_score, scores, None

# ─── callback do sounddevice (roda em thread separada) ───────────────────────

def audio_callback(indata: np.ndarray, frames: int, time, status):
    if not state.capturing or not state.clients:
        return

    chunk = indata[:, 0]   # mono

    # Processamento de sinal básico
    rms      = compute_rms(chunk)
    waveform = compute_waveform(chunk)

    # Detecção de Wake Word
    wake_detected, wake_name, wake_score, wake_scores, wake_error = predict_wakeword(chunk)

    payload = json.dumps({
        "rms":        round(rms, 4),
        "waveform":   [round(v, 4) for v in waveform],
        "frames":     frames,
        "wake_word":  wake_detected,
        "wake_name":  wake_name,
        "wake_score": round(wake_score, 4),
        "wake_scores": {name: round(score, 4) for name, score in wake_scores.items()},
        "wake_models": WAKEWORD_NAMES,
        "wake_error": wake_error,
    })

    # envia para todos os clientes conectados a partir do event loop async
    if state.loop and state.loop.is_running():
        asyncio.run_coroutine_threadsafe(_broadcast(payload), state.loop)

async def _broadcast(payload: str):
    dead = []
    for ws in state.clients:
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        state.clients.remove(ws)

# ─── stream de áudio (singleton) ─────────────────────────────────────────────

_stream: sd.InputStream | None = None

def start_stream():
    global _stream
    if _stream is not None:
        return
    _stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        blocksize=CHUNK_SIZE,
        callback=audio_callback,
    )
    _stream.start()

def stop_stream():
    global _stream
    if _stream is not None:
        _stream.stop()
        _stream.close()
        _stream = None

# ─── rotas HTTP ──────────────────────────────────────────────────────────────

@app.on_event("startup")
async def on_startup():
    state.loop = asyncio.get_running_loop()
    start_stream()   # stream sempre aberto; captura controlada por state.capturing

@app.on_event("shutdown")
async def on_shutdown():
    stop_stream()

@app.get("/")
async def index():
    return FileResponse("index.html")

@app.post("/capture/start")
async def capture_start():
    state.capturing = True
    return {"status": "capturing"}

@app.post("/capture/stop")
async def capture_stop():
    state.capturing = False
    return {"status": "stopped"}

@app.get("/devices")
async def list_devices():
    devices = sd.query_devices()
    inputs  = [
        {"index": i, "name": d["name"], "channels": d["max_input_channels"]}
        for i, d in enumerate(devices)
        if d["max_input_channels"] > 0
    ]
    return {"devices": inputs, "default": sd.default.device[0]}

@app.get("/wakeword")
async def wakeword_info():
    return {
        "models": WAKEWORD_NAMES,
        "threshold": WAKEWORD_THRESHOLD,
    }

# ─── WebSocket ────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    state.clients.append(ws)
    try:
        while True:
            await ws.receive_text()   # mantém a conexão viva
    except WebSocketDisconnect:
        if ws in state.clients:
            state.clients.remove(ws)

# ─── entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
