"""

Usage:
    python param_info.py -p /home/nguyen/Documents/intern20/models_FEo_pooling_VISPEL
"""

import os
import _init_paths
import argparse
from analysis.params import check_param_situ


def argument_parser():
    """
    Create a parser with some common arguments.

    Returns:
        argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', "--path", required=True, help="path to trained models")

    return parser


def main():
    """

    Returns
    -------
    """
    args = argument_parser().parse_args()
    # list of situations within corresponding detector
    mobi_results, rcnn_results = check_param_situ(args.path, 'SOLVER', 'PFT')
    print('MOBI')
    print(mobi_results)
    print('RCNN')
    print(rcnn_results)


if __name__ == '__main__':
    main()
