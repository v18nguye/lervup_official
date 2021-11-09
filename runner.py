"""Runner for reexperiencing the methods
and its results published in the paper.

usage:
    python runner.py --method base
"""
import os
import sys
sys.path.append('./lervup_algo/')
sys.path.append('./lervup_algo/lib/')
sys.path.append('./lervup_algo/vispel/')
import argparse
from base_algo import run_base_opt
from lervup_algo import run_lervup

import warnings
warnings.filterwarnings("ignore")


def argument_parser():
    """
    Create a parser with some common arguments.

    returns:
        argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=['base_opt', 'lervup_fr'], help="pretrained-model settings")

    return parser


def runner(method):
    """Runner ...
    
    :param method: str
        user's visual exposure learning method.
            - base_opt: the baseline with optimal threshold per obj and detector selection.
            - lervup: lervup + focal rating.

    """

    if method == 'base_opt':
        run_base_opt()
        print('\n')

    if method == 'lervup_fr':
        run_lervup('./lervup_algo/lervup/fr/')
        print('\n')
         

if __name__ == '__main__':
    args = argument_parser().parse_args()
    runner(args.method)
