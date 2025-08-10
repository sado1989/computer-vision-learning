import numpy as np
from cv_guide.utils import normalize_img

def test_normalize_img():
    arr = np.array([0, 1, 2], dtype=np.float32)
    out = normalize_img(arr)
    assert np.isclose(out.min(), 0.0)
    assert np.isclose(out.max(), 1.0)
