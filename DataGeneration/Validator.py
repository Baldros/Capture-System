import numpy as np
import soundfile as sf

def validate_audio(path: str) -> tuple[bool, dict]:
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    zcr = np.mean(np.abs(np.diff(np.sign(audio))) > 0)

    frame_size = int(sr * 0.025)
    spectral_vars = []
    for i in range(0, len(audio) - frame_size, frame_size):
        frame = audio[i:i+frame_size] * np.hanning(frame_size)
        spectrum = np.abs(np.fft.rfft(frame))
        if spectrum.max() > 0:
            spectrum = spectrum / spectrum.max()
        spectral_vars.append(np.var(spectrum))

    spectral_var = float(np.median(spectral_vars))

    metrics = {
        "zcr": float(zcr), # zero crossing rate
        "spectral_var": spectral_var,
    }

    passed = zcr > 0.008

    return passed, metrics