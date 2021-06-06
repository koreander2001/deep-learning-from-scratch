from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class Perceptron:
    W: np.ndarray
    b: float
    activation_func: Callable[[np.ndarray], np.ndarray]

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.activation_func(np.dot(X, self.W) + self.b)  # type: ignore
