"""

Baseline training with Focal Rating
Usage:
    
run with default settings:
    python3 train_test_basefr.py --config_file ./configs/mobi_BL.yaml --model_name bank_mobi.pkl --situation BANK
    
run with more customized settings:    
    python3 train_test_basefr.py --config_file ./configs/mobi_BL.yaml --model_name bank_mobi_cv.pkl --situation BANK --opts SOLVER.CROSS_VAL False FE.K 10 FE.GAMMA 2 OUTPUT.DIR ./mobinet_models/

"""
import os
import sys
sys.path.append('./lib/')
import pickle
import json
import argparse
import numpy as np
from lib.baseline.baseline import BaseFocalRating as BFR
from lib.config.config import get_cfg


def argument_parser():
    """
    Create a parser with some common arguments.

    Returns:
        argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--model_name", required=True, help="saved model name")
    parser.add_argument("--situation", required=True, help="IT, ACCOM, BANK, WAIT")
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )

    return parser


def save_model(model, filename, out_dir):
    out_dir_path = out_dir
    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path)

    out_file_path = os.path.join(out_dir_path, filename)

    with open(out_file_path, 'wb') as output:
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)

    with open(model.save_path, 'w') as fp:
        json.dump(model.opt_thresholds, fp)


def setup(args):
    """

    :param args:
    :return:
        cfg
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    return cfg


def main():
    """

    return
    -------

    """
    args = argument_parser().parse_args()
    cfg = setup(args)

    model = BFR(cfg, args.situation, args.model_name)
    model.optimize()

    if cfg.OUTPUT.VERBOSE:
        print("Saved model !")

    save_model(model, args.model_name, cfg.OUTPUT.DIR)
    print('Best Model Corr: '+ "{:.4f}".format(model.test_result))

if __name__ == "__main__":
    main()
