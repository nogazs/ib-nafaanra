import os
from utils.project import get_config, get_logger, Pipeline
from ib_color_naming.src.ib_naming_model import load_model
from analysis import optimality, random_drift, plots
from data.make_dataset import make_dataset

curr_path = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    config = get_config(curr_path + '/config.json')
    logger = get_logger(config.name)
    logger.info('loading wcs plus dataset')
    wcsp = make_dataset()
    logger.info('loading IB model')
    IB_model = load_model(model_dir=curr_path + '/../models/')
    proj_kwargs = {
        'config': config,
        'wcsp': wcsp,
        'IB_model': IB_model
    }

    analyses = [
        optimality.wcsp_scores,
        optimality.eval_nafaanra_systems,
        optimality.rotation_analysis,
        random_drift.sim_rand_drift,
        random_drift.traj_scores
    ]

    figures = [
        plots.nafaanra_data,
        plots.nafaanra18_spkr_maps,
        plots.info_plane_nafaanra,
        plots.rotations_fig,
        plots.info_plane_traj,
        plots.random_drift,
        plots.english_interpolation,
    ]

    pipeline = Pipeline(analysis_funcs=analyses, plot_funcs=figures, proj_kwargs=proj_kwargs, **config.__dict__)
    # pipeline.run()
    pipeline.generate_figs(figures)
