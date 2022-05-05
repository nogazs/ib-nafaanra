import numpy as np
from scipy.special import logsumexp

PRECISION = 1e-16


def marginal(pXY, axis=1):
    """:return pY (axis = 0) or pX (default, axis = 1)"""
    return pXY.sum(axis)


def conditional(pXY):
    """:return  pY_X """
    pX = pXY.sum(axis=1, keepdims=True)
    return np.where(pX > PRECISION, pXY / pX, 1 / pXY.shape[1])


def joint(pY_X, pX):
    """:return  pXY """
    return pY_X * pX[:, None]


def marginalize(pY_X, pX):
    """:return  pY """
    return pY_X.T @ pX


def normalize_rows(A, p0=None):
    if len(A.shape) == 1:
        return A/A.sum()
    Z = A.sum(axis=1)[:, None]
    p0 = p0 if p0 is not None else 1/A.shape[1]
    return np.where(Z > 0, A / A.sum(axis=1)[:, None], p0)


def bayes(pY_X, pX):
    """:return pX_Y """
    if len(pX.shape) == 2 and pX.shape[1] == 1:
        pX = pX[:, 0]
    pXY = joint(pY_X, pX)
    pY = pXY.sum(axis=0)[:, None]
    return np.where(pY > PRECISION, pXY.T / pY, 1 / pXY.shape[0])


def softmax(dxy, beta=1, axis=None):
    """:return
        axis = None: pXY propto exp(-beta * dxy)
        axis = 1: pY_X propto exp(-beta * dxy)
        axis = 0: pX_Y propto exp(-beta * dxy)
    """
    log_z = logsumexp(-beta * dxy, axis, keepdims=True)
    return np.exp(-beta * dxy - log_z)

