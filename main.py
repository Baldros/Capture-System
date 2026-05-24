import asyncio
import json
import math
import threading

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
oww_model = Model(wakeword_models=["alexa"], inference_framework="onnx")

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

# ─── callback do sounddevice (roda em thread separada) ───────────────────────

def audio_callback(indata: np.ndarray, frames: int, time, status):
    if not state.capturing or not state.clients:
        return

    chunk = indata[:, 0]   # mono

    # Processamento de sinal básico
    rms      = compute_rms(chunk)
    waveform = compute_waveform(chunk)

    # Detecção de Wake Word
    prediction = oww_model.predict(chunk)
    wake_score = float(prediction.get("alexa", 0.0))
    wake_detected = bool(wake_score > 0.5)

    payload = json.dumps({
        "rms":        round(rms, 4),
        "waveform":   [round(v, 4) for v in waveform],
        "frames":     frames,
        "wake_word":  wake_detected,
        "wake_score": round(wake_score, 4)
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