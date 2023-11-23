import numpy as np


def framing(data: np.ndarray, frame_size: int, frame_shift: int) -> np.ndarray:
    if data.ndim > 2:
        raise ValueError(
            f"window data, expected input shape is 2-D or 1-D, but input shape is {data.shape}"
        )
    full_frame_size = data.shape[-1]

    # if size is underfitted, then pad zero value
    if (pad_size := (full_frame_size - frame_size) % frame_shift) != 0:
        if frame_shift > frame_size:
            pad_size = frame_shift + frame_size - pad_size
        else:
            pad_size = frame_size - pad_size
        pad = np.zeros((data.shape[0], pad_size) if data.ndim == 2 else (pad_size,))
        data = np.concatenate((data, pad), axis=-1)
        print(f"windowing size: padding zero-data, shape {pad.shape}")

    # shift_n = data.shape[-1] // frame_shift
    # usually frame size is much smaller then shift count, so iteration with frame size
    if data.ndim == 2:
        data = np.stack(
            [
                data[:, i : data.shape[-1] + i - frame_size + 1 : frame_shift]
                for i in range(frame_size)
            ],
            axis=2,
        )
    else:
        data = np.stack(
            [
                data[i : data.shape[-1] + i - frame_size + 1 : frame_shift]
                for i in range(frame_size)
            ],
            axis=1,
        )
    return data
