from __future__ import annotations

from scipy.io.wavfile import read
import numpy as np
from enum import Enum
from typing import Tuple


class RawVoiceData:
    def __init__(self, path: str) -> None:
        """Read voice file from path

        Args:
            path (str): voice file path
        """
        sample_rate, data = read(path)
        self.sample_rate = sample_rate
        self.is_stereo = True
        if data.ndim == 2:
            self.is_stereo = True
            self.data = data.T
        else:
            self.is_stereo = False
            self.data = np.array([data, data])

    def get_voice_data(self) -> Tuple[VoiceData, VoiceData]:
        return VoiceData(self.data[0], self.sample_rate), VoiceData(
            self.data[1], self.sample_rate
        )

    def __len__(self) -> int:
        return self.data.shape[-1]


class VoiceData:
    def __init__(self, data: np.ndarray, sample_rate: int) -> None:
        self.data = data
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return self.data.shape[0]
