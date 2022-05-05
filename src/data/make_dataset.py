from .wcs import WCSPlus
from .nafaanra2018 import add_nafaanra_2018


def make_dataset(data_file=None):
    wcsp = WCSPlus(data_file=data_file)
    add_nafaanra_2018(wcsp)
    return wcsp
