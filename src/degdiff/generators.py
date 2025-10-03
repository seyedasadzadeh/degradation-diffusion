"""Small generator module extracted from learner.ipynb

Only implements the ParisLawDegradation used in the notebook.
"""
import numpy as np


class BaseDegradationProcess:
    def __init__(self, length, dim):
        self.length = int(length)
        self.dim = int(dim)

    def generate_episode(self, x0):
        x0 = np.atleast_1d(np.asarray(x0))
        episode = np.zeros((x0.shape[0], self.length + 1))
        episode[:, 0] = x0
        for i in range(self.length):
            episode[:, i + 1] = episode[:, i] + self.xdot(episode[:, i])
        return episode


class ParisLawDegradation(BaseDegradationProcess):
    """Paris–Erdogan fatigue crack growth model.

    Parameters mirrored from the notebook: C, m, delta_sigma, beta.
    """

    def __init__(self, length, dim, C=1e-12, m=3, delta_sigma=100, beta=1):
        super().__init__(length, dim)
        self.C = float(C)
        self.m = float(m)
        self.delta_sigma = float(delta_sigma)
        self.beta = float(beta)

    def delta_K(self, a):
        a = np.atleast_1d(np.asarray(a))
        return self.delta_sigma * np.sqrt(np.pi * a) * self.beta

    def xdot(self, a):
        a = np.atleast_1d(np.asarray(a))
        return self.C * (self.delta_K(a) ** self.m)
