"""Global seed helper produces stable NumPy draws (used with estimator ``random_state``)."""

from compassmind.seed import set_global_seed


def test_set_global_seed_is_deterministic_for_numpy():
    set_global_seed(123)
    import numpy as np

    a = np.random.randn(5)
    set_global_seed(123)
    b = np.random.randn(5)
    assert (a == b).all()
