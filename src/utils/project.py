from src.utils.files import *
import sys
from tdict import Tdict
import numpy as np
import logging

ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(name)s - %(funcName)s] %(message)s')
ch.setFormatter(formatter)


def get_logger(logger_name, level='debug'):
    if level == 'info':
        ch.setLevel(logging.INFO)
    elif level == 'warn':
        ch.setLevel(logging.WARN)
    else:
        ch.setLevel(logging.DEBUG)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    return logger


def get_config(config_file):
    config_json = read_json(config_file)
    return Tdict(config_json)


class Results(Tdict):

    def __init__(self, name='res', logger=None):
        super().__init__()
        self.name = name
        if logger is not None:
            self.logger = logger
        else:
            self.logger = get_logger(name)

    def __str__(self):
        self_str = 'Name: %s\nItems:\n' % self.name
        for k, v in sorted(self.items()):
            if k[0] not in ['logger', 'name']:
                if type(v) is np.ndarray:
                    self_str += '%s %snp.array %s\n' % (k, '\t', v.shape)
                else:
                    self_str += '%s %s%s\n' % (k, '\t', type(v).__name__)
        return self_str

    def get_fname(self, path):
        fname = path + self.name
        if self.name != 'res':
            fname += '_res'
        return fname + '.pkl'

    def save(self, path):
        try:
            fname = self.get_fname(path)
            self.logger.info('saving results to %s' % fname)
            logger = self.pop('logger')
            save_obj(fname, self)
            self.logger = logger
        except:
            self.logger.error('saving res failed! %s' % (sys.exc_info()[0]))

    def load(self, path):
        try:
            fname = self.get_fname(path)
            self.logger.info('loading results from %s' % fname)
            res = load_obj(fname)
            self |= res
        except:
            self.logger.error('loading results failed! %s' % (sys.exc_info()[0]))

    def print_res_info(self):
        self.logger.info('===================================')
        self.logger.info('RESULTS INFO')
        self.logger.info('===================================\n' + self.__str__())
        self.logger.info('===================================')


class Pipeline(object):

    def __init__(self, out_dir='', analysis_funcs=[], plot_funcs=[], proj_kwargs=None,
                 name='', warm_start=False, fig_fmt='pdf', seed=None, **kwargs):
        self.out_path = out_dir
        ensure_dir(out_dir)
        self.analysis_funcs = analysis_funcs
        self.plot_funcs = plot_funcs
        self.proj_kwargs = proj_kwargs
        self.name = name
        self.logger = get_logger(name + ' - pipeline')
        self.fig_fmt = '.' + fig_fmt
        self.seed = seed
        if seed is not None:
            self.set_rand_seed(seed)
        self.res = Results(name, logger=self.logger)
        if warm_start:
            self.logger.info('warm start initialization')
            self.load_results()

    def load_results(self):
        self.res.load(self.out_path)

    def save_results(self):
        self.res.save(self.out_path)

    def run(self, analysis_funcs=None, plot_funcs=None, inter_save=True):
        self.logger.info('running full pipeline')
        self.run_analyses(analysis_funcs, inter_save)
        self.generate_figs(plot_funcs)
        self.save_results()
        self.logger.info('pipeline done!')

    def run_analyses(self, analysis_funcs=None, inter_save=True):
        analysis_funcs = self.analysis_funcs if analysis_funcs is None else analysis_funcs
        self.logger.info('running analyses...')
        for func in analysis_funcs:
            self.logger.info('analysis: %s' % func.__name__)
            func(self.res, **self.proj_kwargs)
            if inter_save:
                self.save_results()
        self.logger.info('analyses done')

    def generate_figs(self, plot_funcs=None):
        plot_funcs = self.plot_funcs if plot_funcs is None else plot_funcs
        self.logger.info('exporting figures to: %s' % self.out_path)
        for plot_func in plot_funcs:
            func_name = plot_func.__name__
            self.logger.info('figure: %s' % func_name)
            fig_name = self.out_path + func_name + self.fig_fmt
            plot_func(self.res, fig_name, **self.proj_kwargs)
        self.logger.info('export figs done')

    def set_fig_fmt(self, fig_fmt):
        self.fig_fmt = '.' + fig_fmt

    def set_rand_seed(self, seed):
        self.logger.info('setting numpy random seed %d' % seed)
        self.seed = seed
        np.random.seed(seed)
