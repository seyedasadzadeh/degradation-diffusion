import numpy as np
from degdiff import ParisLawDegradation


def test_paris_generate_shape_and_monotonicity():
    n0 = 100
    gen = ParisLawDegradation(length=50, dim=1, C=1e-8)
    x0 = np.abs(np.random.randn(n0)) * 1e-3 + 1e-4
    episodes = gen.generate_episode(x0)
    # shape: (n0, length+1)
    assert episodes.shape == (n0, 51)
    # monotonic non-decreasing crack length over cycles
    diff = episodes[:, 1:] - episodes[:, :-1]
    assert np.all(diff >= -1e-12)
