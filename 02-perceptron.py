import numpy as np

from lib.activation_functions import step_function
from lib.model import Perceptron


def and_gate(X: np.ndarray) -> np.ndarray:
    perceptron = Perceptron(
        W=np.array([1.0, 1.0]), b=-1.0, activation_func=step_function
    )
    return perceptron.transform(X)


def nand_gate(X: np.ndarray) -> np.ndarray:
    perceptron = Perceptron(
        W=np.array([-1.0, -1.0]), b=1.5, activation_func=step_function
    )
    return perceptron.transform(X)


def or_gate(X: np.ndarray) -> np.ndarray:
    perceptron = Perceptron(
        W=np.array([1.0, 1.0]), b=-0.5, activation_func=step_function
    )
    return perceptron.transform(X)


def xor_gate(X: np.ndarray) -> np.ndarray:
    return and_gate(np.hstack([or_gate(X), nand_gate(X)]))


def main() -> None:
    flag_patterns = [
        np.array([0, 0]),
        np.array([0, 1]),
        np.array([1, 0]),
        np.array([1, 1]),
    ]

    print("and")
    for flag_pattern in flag_patterns:
        print(f"input: {flag_pattern}, output: {and_gate(flag_pattern)}")

    print("nand")
    for flag_pattern in flag_patterns:
        print(f"input: {flag_pattern}, output: {nand_gate(flag_pattern)}")

    print("or")
    for flag_pattern in flag_patterns:
        print(f"input: {flag_pattern}, output: {or_gate(flag_pattern)}")

    print("xor")
    for flag_pattern in flag_patterns:
        print(f"input: {flag_pattern}, output: {xor_gate(flag_pattern)}")


if __name__ == "__main__":
    main()
