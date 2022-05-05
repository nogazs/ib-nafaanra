"""
API for the World Color Survey (WCS) dataset

The WCS dataset is available at:
http://www1.icsi.berkeley.edu/wcs/data.html

--------------------------
author: Noga Zaslavsky
created: November 2016
--------------------------
"""

import string
import numpy as np
import pandas as pd
import os
import pickle
from utils.project import get_logger
from utils.files import ensure_dir, ensure_file
from tdict import Tdict
from ib_color_naming.src.tools import lab2rgb


logger = get_logger('wcs_dataset')

N_CHIPS = 330
N_COLS = 41
N_ROWS = 10
SPACE = 0.1
ROWS = [string.ascii_uppercase[i] for i in range(10)]

# init module
curr_path = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = curr_path + '/../../data/raw/'
DATA_PROCESSED_DIR = curr_path + '/../../data/processed/'
ensure_dir(curr_path + '/../../data/')
ensure_dir(DATA_RAW_DIR)
ensure_dir(DATA_PROCESSED_DIR)
ensure_file(DATA_RAW_DIR + 'cnum-vhcm-lab-new.txt', 'http://www1.icsi.berkeley.edu/wcs/data/cnum-maps/cnum-vhcm-lab-new.txt')
ensure_file(DATA_RAW_DIR + 'chip.txt', 'http://www1.icsi.berkeley.edu/wcs/data/20021219/txt/chip.txt')
ensure_file(DATA_RAW_DIR + 'term.txt', 'http://www1.icsi.berkeley.edu/wcs/data/20021219/txt/term.txt')
ensure_file(DATA_RAW_DIR + 'langs_info.txt', 'https://www1.icsi.berkeley.edu/wcs/data/20021219/txt/lang.txt')

terms_df = pd.read_csv(DATA_RAW_DIR + 'term.txt', delimiter='\t', header=None, keep_default_na=False, na_values=['NaN'])
chips_df = pd.read_csv(DATA_RAW_DIR + 'cnum-vhcm-lab-new.txt', delimiter='\t').sort_values(by='#cnum')
cnums_df = pd.read_csv(DATA_RAW_DIR + 'chip.txt', delimiter='\t', header=None).values

WCS_CHIPS = chips_df[['L*', 'a*', 'b*']].values
WCS_CHIPS_RGB = lab2rgb(WCS_CHIPS)
LANGS = pd.read_csv(DATA_RAW_DIR + 'langs_info.txt', delimiter='\t', header=None, encoding='utf8')
CNUMS_WCS_COR = dict(
    zip(cnums_df[:, 0], [(ROWS.index(cnums_df[cnum - 1, 1]), cnums_df[cnum - 1, 2]) for cnum in cnums_df[:, 0]]))
_WCS_COR_CNUMS = dict(zip(cnums_df[:, 3], cnums_df[:, 0]))


def cnum2ind(cnum):
    """
    convert chip number to location in the WCS palette
    Example: cnum2ind(100) returns (2,22)
    """
    return CNUMS_WCS_COR[cnum]


def code2cnum(code):
    """
    convert WCS palette code to chip number
    Example: code2cnum('C22') returns 100
    :param c: string
    :return:
    """
    if code[0] == 'A':
        return _WCS_COR_CNUMS['A0']
    if code[0] == 'J':
        return _WCS_COR_CNUMS['J0']
    return _WCS_COR_CNUMS[code]


def code2ceilab(code):
    """
    get CIELAB coordinates for chip c
    :param c: string
    :return: np.array (3,)
    """
    return WCS_CHIPS[code2cnum(code)]


def get_achrom_chips():
    A2Z = string.ascii_uppercase
    row_chips = [None] * 10
    for i in range(10):
        row_chips[i] = code2cnum(A2Z[i] + '0') - 1
    return np.array(row_chips, dtype='int')


def get_row_chips(row):
    row_chips = [None] * 40
    for i in range(40):
        row_chips[i] = code2cnum('%s%d' % (row, i + 1)) - 1
    return np.array(row_chips, dtype='int')


def get_col_chips(col):
    rows = ROWS
    col_chips = [None] * 8 if col > 0 else [None] * 10
    for i in range(len(col_chips)):
        col_chips[i] = code2cnum('%s%d' % (rows[i + int(col > 0)], col)) - 1
    return np.array(col_chips, dtype='int')


class WCSPlus:
    """
    API for naming and focus data collected for the WCS color grid. By default, only the WCS data are loaded.
    Additional languages can be added if the data is compatible with the WCS color grid.
    """

    def __init__(self, data_file=None):
        self.chips = WCS_CHIPS
        self.data_file = data_file if data_file is not None else DATA_PROCESSED_DIR + 'wcsp.pkl'
        self.color_space = 'LAB'
        self.chips_rgb = WCS_CHIPS_RGB
        self.gibson_chips = False
        if os.path.isfile(self.data_file):
            logger.info('initializing WCS data from file %s' % self.data_file)
            with open(self.data_file, 'rb') as f:
                data = pickle.load(f)
                self.__dict__.update(data.__dict__)
        else:
            logger.warning('preprocessed data file not found! loading WCS raw data...')
            self._langs = list(LANGS[0])
            self._lang_names = dict(zip(LANGS[0], LANGS[1]))
            self._names2ind = dict(zip(LANGS[1], LANGS[0]))
            self._lang_countries = dict(zip(LANGS[0], LANGS[2]))
            self._naming_data = process_wcs_naming_data()
            self.save()

    def __contains__(self, lang_name):
        return lang_name in self._lang_names.values()

    def all_langs(self):
        return self._langs.copy()

    def n_langs(self):
        return len(self._langs)

    def n_colors(self):
        return self.chips.shape[0]

    def add_lang(self, name, country, naming_data, lang=None):
        if name in self._lang_names.values() or (lang is not None and lang in self._langs):
            logger.warning('%s not added' % name)
            return self._names2ind[name]
        if lang is None or lang in self._langs:
            lang = np.max(self._langs) + 1
        logger.info('adding %s with lang ID %d' % (name, lang))
        self._langs.append(lang)
        self._lang_names[lang] = name
        self._names2ind[name] = lang
        self._lang_countries[lang] = country
        self._naming_data[lang] = naming_data
        return lang

    def remove_langs(self, langs):
        """
        exclude a list of langs
        :param langs: list
        """
        logger.info('excluding %d languages:' % len(langs))
        for lang in langs:
            if lang in self._langs:
                logger.info('\t%d - %s' % (lang, self.lang_name(lang)))
                name = self._lang_names.pop(lang)
                self._names2ind.pop(name)
                self._lang_countries.pop(lang)
                self._naming_data.pop(lang)
                self._langs.remove(lang)

    def lang_name(self, lang, short=False):
        if short:
            return self._lang_names[lang].split(' (')[0]
        return self._lang_names[lang]

    def lang_country(self, lang):
        return self._lang_countries[lang].split(' (')[0]

    def lang_name2ind(self, name):
        return self._names2ind[name]

    def get_naming_data(self, langs=None):
        if langs is None:
            langs = self._langs
        return {lang: self._naming_data[lang] for lang in langs if lang in self._naming_data}

    def lang_nCW(self, lang):
        return self._naming_data[lang].nCW

    def lang_lex(self, lang):
        return self._naming_data[lang].lex

    def lang_pW_C(self, lang, rotation=0):
        return self._naming_data[lang].pW_C

    def lang_spkr_data(self, lang):
        if hasattr(self._naming_data[lang], 'nCW_spkr'):
            return self._naming_data[lang].nCW_spkr

    def rot_perm(self, rotation):
        perm = np.zeros(N_CHIPS, dtype=int)
        r = rotation if rotation > 0 else 40 + rotation
        for cnum, (row, col) in CNUMS_WCS_COR.items():
            if col > 0:
                col = col + r
                if col > 40:
                    col -= 40
            perm[code2cnum('%s%d' % (ROWS[row], col)) - 1] = cnum - 1
        return perm

    def rotate(self, P, rotation):
        """
        :param P:
        :param rotation: rotation in range(40)
        :return:
        """
        rotated = self.rot_perm(rotation)
        return P[rotated]

    def set_rotation(self, rotation):
        perm = self.rot_perm(rotation)
        for lang in self._naming_data.keys():
            self._naming_data[lang].pW_C = self._naming_data[lang].pW_C[perm]
            self._naming_data[lang].nCW = self._naming_data[lang].nCW[perm]

    def save(self):
        with open(self.data_file, 'wb') as f:
            pickle.dump(self, f)


def get_lex(terms_l):
    lex = sorted(list(terms_l[3].unique()))
    if '*' in lex:
        lex.remove('*')
    return lex


def process_wcs_naming_data():
    naming_data = dict()
    langs = list(LANGS[0])
    for lang in langs:
        lang_data = Tdict()
        terms_l = terms_df[terms_df[0] == lang]
        lang_data.lex = get_lex(terms_l)
        K = len(lang_data.lex)
        lang_data.pW_C = np.zeros((N_CHIPS, K))
        lang_data.nCW = np.zeros((N_CHIPS, K))
        for i, w in enumerate(lang_data.lex):
            f = terms_l[terms_l[3] == w][2].values
            chips_w, counts = np.unique(f, return_counts=True)
            lang_data.nCW[chips_w - 1, i] = counts
        nC = lang_data.nCW.sum(axis=1)[:, None]
        if (lang_data.nCW.sum(axis=1) == 0).any():
            logger.warning('unlabeled chips in language %d' % lang)
            lang_data.pW_C = lang_data.nCW / (nC + 1e-20)
        else:
            lang_data.pW_C = lang_data.nCW / nC
        naming_data[lang] = lang_data
    return naming_data


if __name__ == '__main__':
    wcsp = WCSPlus()
