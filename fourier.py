import numpy as np
from frame import framing
from window import windowing, WindowingFunc


def FFT(data: np.ndarray) -> np.ndarray:
    return np.fft.fft(data)


def stft(
    data: np.ndarray,
    frame_size: int,
    frame_shift: int,
    windowing_method: WindowingFunc | None = None,
):
    framed_data = framing(data, frame_size, frame_shift)
    if windowing_method is not None:
        framed_data = windowing(framed_data, windowing_method)
    return np.fft.rfft(framed_data)
