import numpy as np


def pre_emphasis(data: np.ndarray, alpha=0.97) -> np.ndarray:
    """Pre emphasis raw data
    f[n] - alpha * f[n-1]

    Args:
        data (np.ndarray): 1-D data
        alpha (float, optional): Defaults to 0.97.

    Raises:
        ValueError: data is not 1-D data (shape (m,)), then raise

    Returns:
        np.ndarray: pre empahsis np.ndarray
    """
    if len(data.shape) >= 2:
        raise ValueError(
            f"pre emphasis error: expected value shape is (n,) 1-D, but input shape is {data.shape}"
        )
    pre = data - alpha * np.hstack(([0], data[:-1]))
    return pre
