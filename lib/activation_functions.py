import numpy as np


def step_function(X: np.ndarray) -> np.ndarray:
    return (X > 0).astype(int)
