import numpy as np

def normalize_img(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)
