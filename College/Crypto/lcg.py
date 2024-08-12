from typing import Iterator
from matplotlib import pyplot as plt
import numpy as np


def linear_congruential_generator(m: int, a: int, c: int, seed: int) -> Iterator[int]:
    """
    This generator implements the Linear Congruential Generator algorithm
    :param m: the modulus, a positive integer constant
    :param a: the multiplier, a non-negative integer constant < m
    :param c: the increment, a non-negative integer constant < m
    :param seed: the starting state of the LCG. It is used to initialize the pseudo-random number sequence
    :return: a non-negative integer in [0, m-1] representing the i-th state of the generator
    """
    x = seed
    while True:
        yield x
        x = (a * x + c) % m


def rand_float_samples(n_samples: int, seed: int = 123_456_789) -> list[float]:
    """
    This function uses an LCG to output a sequence of pseudo-random floats from the uniform distribution on [0, 1)
    :param n_samples: the number of pseudo-random floats to generate
    :param seed: the starting state of the LCG. It is used to initialize the pseudo-random number sequence
    :return: a list of length n_samples containing the generated pseudo-random numbers
    """
    m: int = 2_147_483_648
    a: int = 594_156_893
    c: int = 0
    gen = linear_congruential_generator(m, a, c, seed)
    sequence = []

    for i in range(0, n_samples):
        rand: float = next(gen) / m
        sequence.append(rand)

    return sequence


def test_uniformity(samples: list[float]):
    """
    Test if the generated samples are uniformly distributed by plotting a histogram
    :param samples: the list of pseudo-random samples
    """
    plt.hist(samples, bins=20, density=True)
    plt.title("Uniformity Test - Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()


def test_scalability(samples: list[float]):
    """
    Test the scalability by plotting samples against their indices to check for patterns
    :param samples: the list of pseudo-random samples
    """
    plt.scatter(range(len(samples)), samples)
    plt.title("Scalability Test - Scatter Plot")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.show()


def test_consistency(n_tests: int = 5, n_samples: int = 1000, seed: int = 123_456_789):
    """
    Test the consistency by generating multiple sequences and comparing them
    :param n_tests: number of sequences to generate
    :param n_samples: number of samples in each sequence
    :param seed: the seed value for the LCG
    """
    sequences = [rand_float_samples(n_samples, seed) for _ in range(n_tests)]

    plt.figure(figsize=(12, 8))
    for i, seq in enumerate(sequences):
        plt.plot(seq, label=f"Sequence {i + 1}")

    plt.title("Consistency Test - Multiple Sequences")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    n = 1000
    rand_sequence = rand_float_samples(n)

    # Uniformity Test
    test_uniformity(rand_sequence)

    # Scalability Test
    test_scalability(rand_sequence)

    # Consistency Test
    test_consistency()
