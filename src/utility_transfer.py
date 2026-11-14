"""Transfer a bounded learned utility correction to a fixed probability model."""
import numpy as np
from scipy.special import softmax


def utility(p):
    p = np.asarray(p, np.float64).reshape(-1, 10)
    outside = 1 - p.sum(1, keepdims=True)
    assert np.isfinite(p).all() and np.all(p > 0) and np.all(outside > 0)
    return np.log(p) - np.log(outside)


def transfer(baseline, new, source, strength):
    assert 0 <= strength <= 1
    delta = utility(new) - utility(source)
    assert np.max(abs(delta)) <= .50001
    z = utility(baseline) + strength * delta
    return softmax(np.c_[z, np.zeros(len(z))], axis=1)[:, :10].ravel()
