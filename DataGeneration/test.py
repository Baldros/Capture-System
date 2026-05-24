import numpy as np
import soundfile as sf
import os

def diagnosticar(pasta: str):
    for nome in sorted(os.listdir(pasta)):
        if not nome.endswith(".wav"):
            continue
        path = os.path.join(pasta, nome)
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

        spectral_var = float(np.mean(spectral_vars))
        print(f"{nome} | zcr={zcr:.4f} | spectral_var={spectral_var:.5f}")

diagnosticar(r"E:\Capture System\DataGeneration\Dataset-Atlas")