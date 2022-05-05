import matplotlib.pyplot as plt
from src.utils.info import H
from ib_color_naming.src.figures import mode_map
import numpy as np


import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
plt.rcParams['savefig.dpi'] = 300
fig_scale = 1

FS = 13
traj_ex_clr = np.array([72, 148, 139]) / 255
traj_clr = 'gray'


# ~~~~~~~~~~~~~
# plot data
# ~~~~~~~~~~~~~

def nafaanra_data(res, fig_name, IB_model,  **kwargs):
    pM = IB_model.pM  # for visualization only
    figsize = (12, 2.5)
    plt.figure(figsize=np.multiply(figsize, fig_scale).tolist())
    plt.subplot(121)
    mode_map(res.nf1978.pW_C, pM)
    plt.title('A. 1978 system', loc='left', fontsize=20)
    plt.subplot(122)
    plt.title('B. 2018 system', loc='left', fontsize=20)
    mode_map(res.nf2018.pW_C, pM)
    plt.tight_layout()
    plt.savefig(fig_name)


def nafaanra18_spkr_maps(res, fig_name, IB_model, wcsp,  **kwargs):
    pM = IB_model.pM  # for visualization only
    pW_C_spkr = wcsp._naming_data[112].pW_C_spkr
    spkr_ages = wcsp._naming_data[112].spkr_ages
    order = spkr_ages.argsort()
    figsize = (4.04, 6.38)
    plt.figure(figsize=np.multiply(figsize, fig_scale).tolist())
    for i, pi in enumerate(order):
        plt.subplot(8, 2, i+1)
        plt.title('Participant {}, age {}'.format(i+1, spkr_ages[pi]))
        mode_map(pW_C_spkr[pi], pM)
    plt.tight_layout(w_pad=1)
    plt.savefig(fig_name)


# ~~~~~~~~~~~~~
# info planes
# ~~~~~~~~~~~~~

def info_plane(IB_model, ax=None):
    if ax is None:
        figsize = (4.9, 3.5)
        fig = plt.figure(figsize=np.multiply(figsize, fig_scale).tolist())
        ax = plt.gca()
    I_MU = IB_model.I_MU
    Ix, Iy = IB_model.IB_curve
    HM = H(IB_model.pM[:, 0])
    plt.plot(Ix, Iy, linewidth=3, color='k', label='Theoretical limit')
    plt.xlabel('Complexity (bits)', fontsize=12)
    plt.ylabel('Accuracy (bits)', fontsize=12)
    plt.xlim([0, HM])
    plt.ylim([0, I_MU + .5])
    return ax


def info_plane_nafaanra(res, fig_name, IB_model, wcsp, **kwargs):
    info_plane(IB_model)

    # WCS data
    for l, l_data in wcsp.get_naming_data().items():
        pW_C = l_data.pW_C
        plt.plot(IB_model.complexity(pW_C), IB_model.accuracy(pW_C), 'o', color='lightblue', zorder=0)
    plt.plot(IB_model.complexity(pW_C), IB_model.accuracy(pW_C), 'o', color='lightblue', zorder=0, label='WCS+ languages')

    # Nafaanra systems
    nf_kwargs = {'mec': 'k', 'clip_on': False, 'zorder': 2}
    pW_C_78 = res.nf1978.pW_C
    pW_C_18 = res.nf2018.pW_C
    plt.plot(IB_model.complexity(pW_C_78), IB_model.accuracy(pW_C_78), 'o', label='Nafaanra 1978', **nf_kwargs)
    plt.plot(IB_model.complexity(pW_C_18), IB_model.accuracy(pW_C_18), 'o', label='Nafaanra 2018', **nf_kwargs)

    plt.text(.5, 0.95 * IB_model.I_MU, 'unachievable', weight=None, fontsize=12)
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    plt.savefig(fig_name)


def info_plane_traj(res, fig_name, IB_model, config, wcsp, example_traj=None, **kwargs):
    info_plane(IB_model)

    # Nafaanra systems
    nf_kwargs = {'mec': 'k', 'clip_on': False, 'zorder': 2}
    pW_C_78 = res.nf1978.pW_C
    pW_C_18 = res.nf2018.pW_C
    plt.plot(IB_model.complexity(pW_C_78), IB_model.accuracy(pW_C_78), 'o', label='Nafaanra 1978', **nf_kwargs)
    plt.plot(IB_model.complexity(pW_C_18), IB_model.accuracy(pW_C_18), 'o', label='Nafaanra 2018', **nf_kwargs)

    if 'English' in wcsp._langs:
        pW_C_eng = wcsp.lang_pW_C(111)
        plt.plot(IB_model.complexity(pW_C_eng), IB_model.accuracy(pW_C_eng), 'xk', markersize=3.5, label='English', zorder=2)

    # trajectories
    info_trajs = res.rand_drift.info_trajs
    for traj in info_trajs:
        plt.plot(traj[0], traj[1], lw=2, color=traj_clr, zorder=0)
    if example_traj is None:
        example_traj = config.rand_drift.examples[0]
    traj = info_trajs[example_traj]
    plt.plot(traj[0], traj[1], lw=.7, color=traj_ex_clr, zorder=1)
    plt.plot(traj[0, -1], traj[1, -1], '-', color=traj_ex_clr, mec='k', label='Random drift')
    plt.plot(traj[0, -1], traj[1, -1], 'p', color=traj_ex_clr, mec='k')

    plt.text(.8, 0.8 * IB_model.I_MU, 'unachievable', weight=None, fontsize=12)
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    plt.savefig(fig_name)


def english_interpolation(res, fig_name, IB_model, config, wcsp, **kwargs):

    if 'English' not in wcsp._langs:
        return

    info_plane(IB_model)
    # Nafaanra systems
    nf_kwargs = {'mec': 'k', 'clip_on': False, 'zorder': 2}
    pW_C_78 = res.nf1978.pW_C
    pW_C_18 = res.nf2018.pW_C
    pW_C_eng = wcsp.lang_pW_C(111)
    plt.plot(IB_model.complexity(pW_C_78), IB_model.accuracy(pW_C_78), 'o', label='Nafaanra 1978', **nf_kwargs)
    plt.plot(IB_model.complexity(pW_C_18), IB_model.accuracy(pW_C_18), 'o', label='Nafaanra 2018', **nf_kwargs)
    plt.plot(IB_model.complexity(pW_C_eng), IB_model.accuracy(pW_C_eng), 'o', label='English', **nf_kwargs)

    pW_C_78_pad = 0 * pW_C_eng
    lex78 = wcsp.lang_lex(77)
    lex_eng = wcsp.lang_lex(111)
    mapping = [lex_eng.index('GRAY'), lex_eng.index('WHITE'), lex_eng.index('RED'), lex_eng.index('BLACK')]

    for w, w_eng in enumerate(mapping):
        pW_C_78_pad[:, w_eng] = pW_C_78[:, w]
        print(lex78[w], lex_eng[w_eng], w_eng)

    alphas = np.linspace(0, 1)
    comp_a = [IB_model.complexity(a * pW_C_78_pad + (1 - a) * pW_C_eng) for a in alphas]
    acc_a = [IB_model.accuracy(a * pW_C_78_pad + (1 - a) * pW_C_eng) for a in alphas]
    plt.plot(comp_a, acc_a, '--k', lw=.8, zorder=0)

    plt.text(.8, 0.8 * IB_model.I_MU, 'unachievable', weight=None, fontsize=12)
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    plt.savefig(fig_name)


# ~~~~~~~~~~~~~
# rotations
# ~~~~~~~~~~~~~

def rot_subfid(lang_data, fig_gridspec, col=0):
    plt.subplot(fig_gridspec[0, col])
    plot_rot_vals(lang_data.rotations.epsilon)
    plt.ylabel('$\Delta$ efficiency loss', fontsize=FS)
    plt.subplot(fig_gridspec[1, col])
    plot_rot_vals(lang_data.rotations.gnid)
    plt.ylabel('$\Delta$ similarity loss', fontsize=FS)
    plt.xlabel('rotation', fontsize=FS, labelpad=-.1)


def plot_rot_vals(x, label=None):
    rot = range(-19, 21)
    ind = np.hstack((range(21, 40), range(21)))
    plt.plot(rot, x[ind] - x[0], '.-', label=label)
    plt.xlim([-19, 20])
    plt.ylim([0, (x - x[0]).max() + .02])
    plt.xticks([-19, -10, 0, 10, 20])


def rotations_fig(res, fig_name, IB_model, **kwargs):
    pM = IB_model.pM
    figsize = (6.56, 5.86)
    fig = plt.figure(figsize=np.multiply(figsize, fig_scale).tolist())
    # outer_grid = fig.add_gridspec(2, 1, height_ratios=[1.75, 3], hspace=.1)
    # rotations = outer_grid[1].subgridspec(2, 2, wspace=.3, hspace=.3)

    rotations = fig.add_gridspec(2, 2, wspace=.3, hspace=.3)
    rot_subfid(res.nf1978, rotations)
    plt.subplot(rotations[0, 0])
    plt.title('1978', fontsize=FS)
    rot_subfid(res.nf2018, rotations, col=1)
    plt.subplot(rotations[0, 1])
    plt.title('2018', fontsize=FS)

    if fig_name is not None:
        rotations.tight_layout(fig, h_pad=1.)
        plt.savefig(fig_name)


# ~~~~~~~~~~~~~
# random drift
# ~~~~~~~~~~~~~

def rand_drift_efficiency(res, fig_name, config, **kwargs):
    if fig_name is not None:
        figsize = (5.2, 3.64)
        plt.figure(figsize=np.multiply(figsize, fig_scale).tolist())
    n_iter = res.rand_drift.el.shape[1]
    x_itr = range(0, n_iter)
    plt.plot(x_itr, res.nf1978.el * np.ones(n_iter), lw=2, label='Nafaanra 1978')
    plt.plot(x_itr, res.nf2018.el * np.ones(n_iter), lw=2, label='Nafaanra 2018')
    plt.plot(x_itr, res.wcsp_scores.el.mean() * np.ones(n_iter), '--k', lw=1, label='WCS+ (average)')
    plt.plot(x_itr, res.rand_drift.el.mean(axis=0), color='k', lw=2, zorder=10, label='random drift (average)')
    plt.plot(x_itr, res.rand_drift.el[config.rand_drift.examples[0]], color=traj_ex_clr,
             zorder=4, label='random drift (example)')
    plt.plot(x_itr, res.rand_drift.el.T, color=traj_clr, zorder=0)
    plt.xlim([0, n_iter-1])
    plt.ylim([0, 1.1])
    plt.xlabel('iteration', fontsize=FS + 2)
    plt.ylabel('inefficiency, $\\varepsilon(t)$', fontsize=FS + 2)
    plt.legend(fontsize=11)
    if fig_name is not None:
        plt.tight_layout()
        plt.savefig(fig_name)


def rand_drift_similarity(res, fig_name, config, **kwargs):
    if fig_name is not None:
        figsize = (5.2, 3.64)
        plt.figure(figsize=np.multiply(figsize, fig_scale).tolist())
    n_iter = res.rand_drift.el.shape[1]
    x_itr = range(0, n_iter)
    plt.plot(x_itr, 1-res.nf1978.gnid * np.ones(n_iter), lw=2, label='Nafaanra 1978')
    plt.plot(x_itr, 1-res.nf2018.gnid * np.ones(n_iter), lw=2, label='Nafaanra 2018')
    plt.plot(x_itr, 1-res.wcsp_scores.gnid.mean() * np.ones(n_iter), '--k', lw=1, label='WCS+ (average)')
    plt.plot(x_itr, 1-res.rand_drift.gnid_wrt_opt.mean(axis=0), color='k', lw=2, zorder=10, label='random drift (average)')
    plt.plot(x_itr, 1-res.rand_drift.gnid_wrt_opt[config.rand_drift.examples[0]], color=traj_ex_clr,
             zorder=4, label='random drift (example)')
    plt.plot(x_itr, 1-res.rand_drift.gnid_wrt_opt.T, color=traj_clr, zorder=1)
    plt.xlim([0, n_iter-1])
    plt.ylim([0, 1.])
    plt.xlabel('iteration', fontsize=FS + 2)
    plt.ylabel('similarity ($1-$gNID)', fontsize=FS + 2)
    if fig_name is not None:
        plt.tight_layout()
        plt.savefig(fig_name)


def random_drift(res, fig_name, config, **kwargs):
    figsize = (10.4, 4)
    plt.figure(figsize=np.multiply(figsize, fig_scale).tolist())
    plt.subplot(1, 2, 1)
    rand_drift_efficiency(res, fig_name=None, config=config)
    plt.title('A. Deviation from optimality (inefficiency)', fontsize=13, loc='left')
    plt.subplot(1, 2, 2)
    rand_drift_similarity(res, fig_name=None, config=config)
    plt.title('B. Similarity to optimal systems', fontsize=13, loc='left')
    if fig_name is not None:
        plt.tight_layout(w_pad=5)
        plt.savefig(fig_name)
