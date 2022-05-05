import numpy as np
from scipy.stats import multivariate_normal
from utils.stats import bayes
from utils.project import get_logger
from tdict import Tdict
import data.wcs as wcs
from ib_color_naming.src.tools import gNID

logger = get_logger('rand drift')


class RandomDrift:

    def __init__(self, mu, L, weights, thresh=0.01, IB_model=None):
        self.mu = mu.copy()
        self.L = L.copy()
        self.weights = weights.copy()
        self.weights[weights <= thresh] = 0
        self.thresh = thresh
        self.K = len(weights)
        self.cats = weights >= 0
        self.IB_model = IB_model
        self.x = wcs.WCS_CHIPS.copy()
        self.nx = self.x.shape[0]
        self.pW_C, self.pW = None, None
        self.update_system()
        self.randi = np.random.randint
        self.info_traj = np.zeros((2, 0))
        self.pW_C_traj = []

    def update_system(self):
        self.cats = self.weights > self.thresh
        k = self.cats.sum()
        if k == 0:
            self.cats = self.weights == self.weights.max()
        pW = self.weights[self.cats]
        self.pW = pW / pW.sum()
        pC_W = np.zeros((k, self.nx))
        for wi, w in enumerate(np.where(self.cats)[0]):
            Sw = self.cov_cat(w)
            pC_W[wi] = multivariate_normal.pdf(self.x, mean=self.mu[w], cov=Sw)
            pC_W[wi] /= pC_W[wi].sum()
        self.pW_C = bayes(pC_W, self.pW)

    def rand_sign(self, size=None):
        return np.sign(self.randi(2, size=size) - 0.5)

    def cov_cat(self, w):
        return self.L[w] @ self.L[w].T

    def move(self, wt, d_weight, d_sigma):
        # move weights
        self.weights[wt] = np.maximum(0, self.weights[wt] + self.rand_sign() * d_weight)
        self.weights /= self.weights.sum()
        if self.cats[wt] and np.random.binomial(1, .5):
            # move mu
            pc_t = multivariate_normal.pdf(self.x, mean=self.mu[wt], cov=self.cov_cat(wt))
            pc_t /= pc_t.sum()
            v = self.x[np.where(np.random.multinomial(1, pc_t))[0][0]]
            self.mu[wt] = (self.mu[wt] + v) / 2
            # move L
            self.L[wt] += np.eye(3) * d_sigma + np.random.randn(3, 3)

    def eval_state(self):
        if self.IB_model is not None:
            complexity = self.IB_model.complexity(self.pW_C)
            accuracy = self.IB_model.accuracy(self.pW_C)
            return np.array([complexity, accuracy])
        return np.array([np.nan, np.nan])

    def simulate(self, T=1000, d_weight=0.01, d_sigma=1):
        pW_C_t = [None] * T
        # init
        if len(self.pW_C_traj) == 0:
            self.pW_C_traj += [self.pW_C.copy()]
            if self.IB_model is not None:
                self.info_traj = self.eval_state()[:, None]
        # simulate
        info_traj = np.zeros((2, T))
        for t in range(T):
            wt = self.randi(self.K)
            self.move(wt, d_weight, d_sigma)
            self.update_system()
            pW_C_t[t] = self.pW_C.copy()
            info_traj[:, t] = self.eval_state()
            if t % 100 == 0:
                logger.info('[%d] k = %d, [Ix, Iy] = %s' % (t, self.cats.sum(), info_traj[:, t]))

        self.info_traj = np.hstack((self.info_traj, info_traj))
        self.pW_C_traj += pW_C_t

        return pW_C_t, info_traj


def rand_diag(lmin, lmax):
    return np.diag((lmax - lmin) * np.random.rand(3) + lmin)


def fit_gaussians(pW_C, pC, K=330):
    pC_W = bayes(pW_C, pC)
    k = pW_C.shape[1]
    weights = np.zeros(K)
    weights[:k] = pW_C.sum(axis=0)
    weights /= weights.sum()
    x = wcs.WCS_CHIPS.copy()
    mu0 = pC_W.dot(x)
    mu = x[np.random.randint(330, size=K)]
    mu[:k] = mu0
    L0 = np.sqrt((pC_W[:, :, None] * ((x - mu[:k, None, :]) ** 2)).sum(axis=1))
    L = np.random.randn(K, 3, 3)
    pC_W_g = np.zeros_like(pC_W)
    for w in range(k):
        L[w] = np.diag(L0[w])
        pC_W_g[w] = multivariate_normal.pdf(x, mean=mu[w], cov=L[w] @ L[w].T)
        pC_W_g[w] /= pC_W_g[w].sum()
    pW_C_g = bayes(pC_W_g, weights[:k])
    for w in range(k, K):
        L[w] += rand_diag(1, 5)
    return mu, L, weights, pW_C_g


def sim_rand_drift(res, IB_model, config, **kwargs):
    pW_C_0 = res.nf1978.pW_C
    pC = IB_model.pM
    mu0, L0, w0, pW_C_g = fit_gaussians(pW_C_0, pC)
    n_traj = config.rand_drift.n_traj
    info_trajs = [None] * n_traj
    pW_C_trj = [None] * n_traj
    logger.info('starting random drift simulation')
    logger.info('number of random drift trajectories to simulate: %d' % n_traj)
    logger.info('trajectory length: %d' % n_traj)
    for t in range(n_traj):
        logger.info('simulation trajectory %d...' % t)
        rand_drift = RandomDrift(mu0, L0, w0, IB_model=IB_model)
        pW_C_trj[t], info_trajs[t] = rand_drift.simulate(T=config.rand_drift.n_iter)
    res.rand_drift = Tdict({'info_trajs': info_trajs, 'pW_C_trj': pW_C_trj})


def traj_scores(res, IB_model, config, **kwargs):
    n_traj = config.rand_drift.n_traj
    n_iter = config.rand_drift.n_iter
    pW_C_trj = res.rand_drift.pW_C_trj
    pC = IB_model.pM
    el = np.zeros((n_traj, n_iter))
    gnid_wrt_opt = np.zeros((n_traj, n_iter))
    gnid_wrt_nf1978 = np.zeros((n_traj, n_iter))
    gnid_wrt_nf2018 = np.zeros((n_traj, n_iter))
    for traj_i, pW_C_i in enumerate(pW_C_trj):
        for iteration, pW_C_it in enumerate(pW_C_i):
            el[traj_i, iteration], gnid_wrt_opt[traj_i, iteration] = IB_model.fit(pW_C_it)[:2]
            gnid_wrt_nf1978[traj_i, iteration] = gNID(res.nf1978.pW_C, pW_C_it, pC)
            gnid_wrt_nf2018[traj_i, iteration] = gNID(res.nf2018.pW_C, pW_C_it, pC)
    res.rand_drift |= Tdict({'el': el,
                             'gnid_wrt_opt': gnid_wrt_opt,
                             'gnid_wrt_nf1978': gnid_wrt_nf1978,
                             'gnid_wrt_nf2018': gnid_wrt_nf2018
                             })


def eval_opt_traj(res, IB_model, **kwargs):
    a = IB_model.fit(res.nf1978.pW_C)[-1]
    b = IB_model.fit(res.nf2018.pW_C)[-1]
    n_betas = b - a + 1
    gnid_wrt_nf1978 = np.zeros(n_betas)
    gnid_wrt_nf2018 = np.zeros(n_betas)
    pC = IB_model.pM
    for bi in range(a, b + 1):
        qW_C = IB_model.qW_M[bi]
        gnid_wrt_nf1978[bi] = gNID(res.nf1978.pW_C, qW_C, pC)
        gnid_wrt_nf2018[bi] = gNID(res.nf2018.pW_C, qW_C, pC)
