import numpy as np
from scipy.fftpack import dct

from fourier import stft
from window import hamming
from filter_bank import create_mel_filter_bank


def mfcc(
    data: np.ndarray, sr: int, L: int = 20, length: int | None = None
) -> np.ndarray:
    if length is None:
        length = sr // 40  # 25ms
    freq_data = stft(data, length, length // 2, hamming)
    mel_filter_bank = create_mel_filter_bank(length // 2 + 1, 0, L)
    bmel = np.dot(mel_filter_bank, abs(freq_data).T)
    mfcc = dct(np.log(bmel), type=3, axis=0, norm="ortho")
    return mfcc
