import numpy as np

EPSILON = 1e-16


def log(v, base=2):
    return np.log(v) / np.log(base)


def xlogx(v, base=2):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(v > EPSILON, v * log(v, base), 0)


def H(p, axis=None, base=2):
    return -xlogx(p, base).sum(axis=axis)


def MI(pXY, base=2):
    return H(pXY.sum(axis=0), base=base) + H(pXY.sum(axis=1), base=base) - H(pXY, base=base)


def VI(pXY, base=2):
    return H(pXY.sum(axis=0), base=base) + H(pXY.sum(axis=1), base=base) - 2 * MI(pXY, base=base)


def DKL(p, q, axis=None, base=2):
    with np.errstate(divide='ignore', invalid='ignore'):
        return (xlogx(p, base=base) - np.where(p > EPSILON, p * log(q + EPSILON, base=base), 0)).sum(axis=axis)


def gNID(pW_X, pV_X, pX):
    if len(pX.shape) == 1:
        pX = pX[:, None]
    elif pX.shape[0] == 1 and pX.shape[1] > 1:
        pX = pX.T
    pXW = pW_X * pX
    pWV = pXW.T.dot(pV_X)
    pWW = pXW.T.dot(pW_X)
    pVV = (pV_X * pX).T.dot(pV_X)
    score = 1 - MI(pWV) / (np.max([MI(pWW), MI(pVV)]))
    return score