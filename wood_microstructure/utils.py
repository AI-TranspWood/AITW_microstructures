import numpy as np
import numpy.typing as npt


def binarize_volume(arr: npt.NDArray, threshold: float = None) -> npt.NDArray:
    """Binarize the volume image if requested"""
    if threshold is None:
        return arr

    res = np.where(arr < threshold, 0, 255).astype(np.uint8)
    return res
