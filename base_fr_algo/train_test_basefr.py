"""

Baseline training with Focal Rating
Usage:
    
run with default settings:
    python3 train_test_basefr.py --config_file ./configs/mobi_BL.yaml --model_name it_mobi.pkl --situation IT --opts OUTPUT.DIR ./base_fr/mobinet/ OUTPUT.VERBOSE True
    python3 train_test_basefr.py --config_file ./configs/mobi_BL.yaml --model_name accom_mobi.pkl --situation ACCOM --opts OUTPUT.DIR ./base_fr/mobinet/ OUTPUT.VERBOSE True
    python3 train_test_basefr.py --config_file ./configs/mobi_BL.yaml --model_name bank_mobi.pkl --situation BANK --opts OUTPUT.DIR ./base_fr/mobinet/ OUTPUT.VERBOSE True
    python3 train_test_basefr.py --config_file ./configs/mobi_BL.yaml --model_name wait_mobi.pkl --situation WAIT --opts OUTPUT.DIR ./base_fr/mobinet/ OUTPUT.VERBOSE True

    python3 train_test_basefr.py --config_file ./configs/rcnn_BL.yaml --model_name it_rcnn.pkl --situation IT --opts OUTPUT.DIR ./base_fr/rcnn/ OUTPUT.VERBOSE True
    python3 train_test_basefr.py --config_file ./configs/rcnn_BL.yaml --model_name accom_rcnn.pkl --situation ACCOM --opts OUTPUT.DIR ./base_fr/rcnn/ OUTPUT.VERBOSE True
    python3 train_test_basefr.py --config_file ./configs/rcnn_BL.yaml --model_name bank_rcnn.pkl --situation BANK --opts OUTPUT.DIR ./base_fr/rcnn/ OUTPUT.VERBOSE True
    python3 train_test_basefr.py --config_file ./configs/rcnn_BL.yaml --model_name wait_rcnn.pkl --situation WAIT --opts OUTPUT.DIR ./base_fr/rcnn/ OUTPUT.VERBOSE True

    
run with more customized settings:    
    python3 train_test_basefr.py --config_file ./configs/mobi_BL.yaml --model_name bank_mobi_cv.pkl --situation BANK --opts SOLVER.CROSS_VAL False FE.K 10 FE.GAMMA 2 OUTPUT.DIR ./mobinet_models/

"""
import os
import sys
import time
sys.path.append('./lib/')
import pickle
import json
import argparse
import numpy as np
from tools.ftune_fr import ftune_model_fr
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

    # save model
    with open(out_file_path, 'wb') as output:
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)

    # save associated optimal thresholds
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
    start = time.time()
    model = BFR(args.situation, args.model_name)
    trained_model, val_result = ftune_model_fr(model, cfg)

    print('total time: ', time.time() - start)
    if cfg.OUTPUT.VERBOSE:
        print("Saved model !")

    save_model(trained_model, args.model_name, cfg.OUTPUT.DIR)
    print('Best Val Corr: '+ "{:.4f}".format(trained_model.test(trained_model.x_val)))

if __name__ == "__main__":
    main()