import numpy as np
import librosa

def load_wav(path, sr=16000):
    wav, _ = librosa.load(path, sr=sr)
    return wav

def melspectrogram(wav, sr=16000, n_fft=800, hop_length=200, win_length=800,
                   n_mels=80, fmin=55, fmax=7600):
    # 输出 (n_mels, T)
    S = librosa.feature.melspectrogram(
        y=wav, sr=sr, n_fft=n_fft, hop_length=hop_length,
        win_length=win_length, n_mels=n_mels, fmin=fmin, fmax=fmax, power=1.0
    )
    S = np.log(np.clip(S, a_min=1e-5, a_max=None))
    return S.astype(np.float32)
