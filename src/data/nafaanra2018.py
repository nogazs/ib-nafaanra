import os
import pandas as pd
import numpy as np
from tdict import Tdict
from src.data.wcs import N_CHIPS, DATA_RAW_DIR
from src.utils.project import get_logger

logger = get_logger('nafaanra2018')


def process_nafaanra_2018_data(data_dir=DATA_RAW_DIR):

    naming_data_file = data_dir + 'nf18_naming_new.csv'
    age_data_file = data_dir + 'nf18_age.csv'

    if os.path.isfile(naming_data_file):
        naming_data_df = pd.read_csv(naming_data_file, delimiter=',', keep_default_na=False, na_values=['NaN'])
    else:
        logger.error('missing naming data file: {}. aborting...'.format(naming_data_file))
        return None

    if os.path.isfile(age_data_file):
        age_data = pd.read_csv(age_data_file, delimiter=',', keep_default_na=False, na_values=['NaN'])
    else:
        logger.warning('missing age data file: {}. continuing without age data...'.format(age_data_file))
        age_data = None

    # fix typos
    naming_data_df = naming_data_df.replace('g', 'fiŋge')
    naming_data_df = naming_data_df.replace('tɔɔrɔ', 'tɔɔnrɔ')
    ignore = {'', 'ø', 'y'}
    lex = set(np.unique(naming_data_df.values))
    lex = list(lex.difference(ignore))

    def spkr_data2map(spkr_data):
        p = np.zeros((N_CHIPS, len(lex)))
        for c, terms in enumerate(spkr_data):
            if terms in lex:
                p[c, lex.index(terms)] += 1
            elif terms not in ignore:
                print('term missing in lex:', terms)
        return p

    nCW = np.zeros((N_CHIPS, len(lex)))
    n_spkrs = len(naming_data_df.keys())
    pW_C_spkr = np.zeros((n_spkrs, N_CHIPS, len(lex)))
    ages = np.zeros(n_spkrs, dtype=int)

    spkr = 0
    for spkr_name, spkr_data in naming_data_df.iteritems():
        if age_data is not None:
            ages[spkr] = age_data[spkr_name]
        p = spkr_data2map(spkr_data)
        nCW += p
        z = p.sum(axis=1)[:, None]
        pW_C_spkr[spkr] = np.where(z > 0, p / z, 1 / len(lex))
        spkr += 1

    nnf18_naming_data = Tdict()
    nnf18_naming_data.lex = lex
    nnf18_naming_data.spkr_ages = ages
    nnf18_naming_data.pW_C = nCW / nCW.sum(axis=1)[:, None]
    nnf18_naming_data.nCW = nCW
    nnf18_naming_data.pW_C_spkr = pW_C_spkr

    return nnf18_naming_data


def add_nafaanra_2018(wcs_data, lang_id=112):
    nnf_name = 'Nafaanra 2018'
    nnf18_naming_data = process_nafaanra_2018_data()
    if nnf18_naming_data is not None:
        if nnf_name in wcs_data._lang_names.values():
            nnf_id = wcs_data.lang_name2ind(nnf_name)
            wcs_data.remove_langs([nnf_id])
        l = wcs_data.add_lang('Nafaanra 2018', 'Ghana', nnf18_naming_data, lang=lang_id)
        wcs_data.save()
        return l
