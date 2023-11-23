import numpy as np
import math
from typing import Callable

WindowingFunc = Callable[[int], np.ndarray]


def hamming(size: int) -> np.ndarray:
    return 0.54 - 0.46 * np.cos(2 * math.pi * np.arange(size) / (size - 1))


def windowing(
    data: np.ndarray, windowing_method: WindowingFunc = hamming
) -> np.ndarray:
    return data * windowing_method(data.shape[-1])
