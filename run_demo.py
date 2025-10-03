"""Small demo runner for the degdiff package.

Runs a tiny generation and prints shapes. Detects available torch device if installed.
"""
from degdiff import ParisLawDegradation
import numpy as np


def main():
    n0 = 1000
    model = ParisLawDegradation(length=100, dim=1, C=1e-8)
    x0 = np.random.randn(n0) * 0.0003 + 0.0045
    episodes = model.generate_episode(x0)
    print("episodes.shape:", episodes.shape)
    print("sample (first row):", episodes[0, :6])


if __name__ == "__main__":
    main()
