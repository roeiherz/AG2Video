import numpy as np


def compute_is(pyx):
    # pyx: p(y|x)
    # py:  p(y)
    p_y = np.mean(pyx, axis=0)
    e = pyx * np.log(pyx / p_y)  # KL divergence between p(y|x) and p(y)
    e = np.sum(e, axis=1)
    e = np.mean(e, axis=0)

    return np.exp(e)
