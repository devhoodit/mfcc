import numpy as np
from typing import Callable


def _create_triangular_scaler_filter_bank(
    f_high: int, f_low: int, L: int, scaler: Callable[[int], int] = lambda x: x
) -> np.ndarray:
    filter_bank = np.zeros((L, f_high))
    freq = np.arange(f_high)
    freq = np.apply_along_axis(scaler, 0, freq)

    for l in range(L):
        sl = l * (scaler(f_high) - scaler(f_low)) / (L + 1) + scaler(f_low)
        sll = (l + 1) * (scaler(f_high) - scaler(f_low)) / (L + 1) + scaler(f_low)
        slll = (l + 2) * (scaler(f_high) - scaler(f_low)) / (L + 1) + scaler(f_low)

        sl_sll = np.where(np.logical_and(freq > sl, freq < sll))
        r_sl = int(np.min(sl_sll))
        l_sll = int(np.max(sl_sll))

        sll_slll = np.where(np.logical_and(freq > sll, freq <= slll))
        r_sll = int(np.min(sll_slll))
        l_slll = int(np.max(sll_slll))

        filter_bank[l][r_sl : l_sll + 1] = (freq[r_sl : l_sll + 1] - sl) / (sll - sl)
        filter_bank[l][r_sll : l_slll + 1] = (freq[r_sll : l_slll + 1] - slll) / (
            sll - slll
        )
    return filter_bank


def create_triangular_filter_bank(f_high: int, f_low: int, L: int, /):
    return _create_triangular_scaler_filter_bank(f_high, f_low, L)


def _mel(x):
    return 2595 * np.log10(1 + x / 700)


def create_mel_filter_bank(f_high: int, f_low: int, L: int, /):
    return _create_triangular_scaler_filter_bank(f_high, f_low, L, _mel)


def create_filter_bank(
    f_high: int, f_low: int, L: int, scaler: Callable[[int], int] = lambda x: x
) -> np.ndarray:
    f_s = f_high * 2
    filter_bank = np.zeros((L, f_high))
    freq = np.arange(f_high)
    freq = np.apply_along_axis(scaler, 0, freq)

    for l in range(L):
        sl = l * (scaler(f_high) - scaler(f_low)) / (L + 1) + scaler(f_low)
        sll = (l + 1) * (scaler(f_high) - scaler(f_low)) / (L + 1) + scaler(f_low)
        slll = (l + 2) * (scaler(f_high) - scaler(f_low)) / (L + 1) + scaler(f_low)

        sl_sll = np.where(np.logical_and(freq > sl, freq < sll))
        r_sl = int(np.min(sl_sll))
        l_sll = int(np.max(sl_sll))

        sll_slll = np.where(np.logical_and(freq > sll, freq <= slll))
        r_sll = int(np.min(sll_slll))
        l_slll = int(np.max(sll_slll))

        filter_bank[l][r_sl : l_sll + 1] = (freq[r_sl : l_sll + 1] - sl) / (sll - sl)
        filter_bank[l][r_sll : l_slll + 1] = (freq[r_sll : l_slll + 1] - slll) / (
            sll - slll
        )
    return filter_bank
