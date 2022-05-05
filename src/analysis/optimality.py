import numpy as np
from tdict import Tdict
from utils.project import get_logger
from utils.info import gNID

logger = get_logger('opt analysis')


def print_lang_res(lang_res):
    logger.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    logger.info('\tLanguage: %s' % lang_res.lang_name)
    logger.info('\tepsilon: %s' % lang_res.el)
    logger.info('\tgNID: %s' % lang_res.gnid)
    logger.info('\tbeta_l: %s' % lang_res.bl)
    logger.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


def get_lang_res(lang_name, wcsp, model, print_res=False):
    lang_res = Tdict()
    lang_res.lang_name = lang_name
    lang_res.lang_id = wcsp.lang_name2ind(lang_name)
    lang_res.pW_C = wcsp.lang_pW_C(lang_res.lang_id)
    lang_res.el, lang_res.gnid, lang_res.bl, lang_res.qW_M = model.fit(lang_res.pW_C)[:4]
    if print_res:
        print_lang_res(lang_res)
    return lang_res


def wcsp_scores(res, IB_model, wcsp, **kwargs):
    wcsp_data = wcsp.get_naming_data(range(1, 112))
    wcsp_el = np.zeros(111)
    wcsp_gnid = np.zeros(111)
    for li, (l, lang_data) in enumerate(wcsp_data.items()):
        wcsp_el[li], wcsp_gnid[li] = IB_model.fit(lang_data.pW_C)[:2]
    res.wcsp_scores = Tdict({'el': wcsp_el, 'gnid': wcsp_gnid})


def eval_nafaanra_systems(res, IB_model, wcsp, **kwargs):
    res.nf1978 = get_lang_res('Nafaanra', wcsp, IB_model, print_res=True)
    res.nf2018 = get_lang_res('Nafaanra 2018', wcsp, IB_model, print_res=True)


def lang_rotation_analysis(lang_res, IB_model, wcsp):
    pW_C = lang_res.pW_C
    complexity = np.zeros(40)
    accuracy = np.zeros(40)
    epsilon = np.zeros(40)
    gnid = np.zeros(40)
    for r in range(40):
        pW_C_r = wcsp.rotate(pW_C, rotation=r)
        complexity[r] = IB_model.complexity(pW_C_r)
        accuracy[r] = IB_model.accuracy(pW_C_r)
        epsilon[r], gnid[r] = IB_model.fit(pW_C_r)[:2]
    rotations = Tdict({'complexity': complexity,
                       'accuracy': accuracy,
                       'epsilon': epsilon,
                       'gnid': gnid})
    lang_res.rotations = rotations


def rotation_analysis(res, IB_model, wcsp, **kwargs):
    lang_rotation_analysis(res.nf1978, IB_model, wcsp)
    lang_rotation_analysis(res.nf2018, IB_model, wcsp)


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
            el[traj_i, iteration], gnid_wrt_opt[traj_i, iteration] = IB_model.scores(pW_C_it)
            gnid_wrt_nf1978[traj_i, iteration] = gNID(res.nf1978.pW_C, pW_C_it, pC)
            gnid_wrt_nf2018[traj_i, iteration] = gNID(res.nf2018.pW_C, pW_C_it, pC)
    res.rand_drift |= Tdict({'el': el,
                             'gnid_wrt_opt': gnid_wrt_opt,
                             'gnid_wrt_nf1978': gnid_wrt_nf1978,
                             'gnid_wrt_nf2018': gnid_wrt_nf2018
                             })


def eval_opt_traj(res, IB_model, **kwargs):
    a = IB_model.fit_ind(res.nf1978.pW_C)[0]
    b = IB_model.fit_ind(res.nf2018.pW_C)[0]
    n_betas = b - a + 1
    gnid_wrt_nf1978 = np.zeros(n_betas)
    gnid_wrt_nf2018 = np.zeros(n_betas)
    pC = IB_model.pM
    for bi in range(a, b+1):
        qW_C = IB_model.qW_M[bi]
        gnid_wrt_nf1978[bi] = gNID(res.nf1978.pW_C, qW_C, pC)
        gnid_wrt_nf2018[bi] = gNID(res.nf2018.pW_C, qW_C, pC)

