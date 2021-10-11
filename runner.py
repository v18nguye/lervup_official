"""Runner for reexperiencing the methods
and its results published in the paper.

usage:
    python runner.py --method base
"""
import argparse
from base_fr_algo import run_basefr
from base_algo import run_base, run_base_opt
from lervup_algo import run_regressor, run_lervup

import warnings
warnings.filterwarnings("ignore")


def argument_parser():
    """
    Create a parser with some common arguments.

    returns:
        argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=['base', 'base_opt', 'base_fr', 'regress',
    'lervup_fr', 'lervup_nofr' ,'lervup_halfobj', 'levrup_halfuser'], help="pretrained-model settings")

    return parser


def runner(method):
    """Runner ...
    
    :param method: str
        user's visual exposure learning method.
            - base: the utmost baseline algo.
            - base_opt: the baseline with optimal threshold per obj and detector selection.
            - base_fr: the baseline with the use of focal rating.
            - regress: raw feature or PCA-reduced feature regression.
            - lervup_nofr: lervup without the focal rating impact.
            - lervup_fr: lervup + focal rating.
            - lervup_halfuser: lervup using only a half of training users.
            - lervup_haflobj: lervup using only a half of visual concepts.

    """
    
    if method == 'base':
        run_base('mobinet')
        print('\n')
        run_base('rcnn')

    if method == 'base_opt':
        run_base_opt('mobinet')
        print('\n')
        run_base_opt('rcnn')

    if method == 'base_fr':
        run_basefr()

    if method == 'regress':
        run_regressor('mobinet')
        print('\n')
        run_regressor('rcnn')

    if method == 'lervup_fr':
        run_lervup('fr')

    if method == 'lervup_nofr':
        run_lervup('no_fr')

    if method == 'lervup_halfobj':
        run_lervup('half_objs')

    if method == 'levrup_halfuser':
        run_lervup('half_users')


if __name__ == '__main__':
    args = argument_parser().parse_args()
    runner(args.method)
