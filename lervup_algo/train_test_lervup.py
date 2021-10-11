"""
The module trains user's exposure predictors using the proposed lervup algo.

run with default settings:

    python train_test_lervup.py --config_file ./configs/mobi_rf_kmeans.yaml --model_name mobi_bank.pkl --situation BANK --verbose True
    python train_test_lervup.py --config_file ./configs/mobi_rf_kmeans.yaml --model_name mobi_accom.pkl --situation ACCOM --verbose True
    python train_test_lervup.py --config_file ./configs/rcnn_rf_kmeans.yaml --model_name rcnn_it.pkl --situation IT --verbose True
    python train_test_lervup.py --config_file ./configs/rcnn_rf_kmeans.yaml --model_name rcnn_wait.pkl --situation WAIT --verbose True

run with more customized settings:

    python3 train_test_lervup.py --config_file ./configs/rf_kmeans_ft_mobi_cv5.yaml --model_name it_mobi.pkl --situation IT --N 200 --opts FE.MODE OBJECT  FE.K 10  FE.GAMMA 2  DETECTOR.LOAD True SOLVER.PFT ORG OUTPUT.DIR ./mobinet_models/

The VISPEL module stands for VISual Photo Exposure Learning.
"""
import os
import sys
sys.path.append('./lib/')
sys.path.append('./vispel/')
import argparse
import pickle
from lib.situ.acronym import situ_decoding
from vispel.config import get_cfg
from vispel.vispel import VISPEL 


def argument_parser():
    """
    Create a parser with some common arguments.

    Returns:
        argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--model_name", required=True, help="saved modeling name")
    parser.add_argument("--situation", required=True, help="IT, ACCOM, BANK, WAIT")
    parser.add_argument("--N", default=-1, type= int, help="Number of training profiles: "
                                                   "-1: ALL user profiles"
                                                   "N: N profiles (N < 400)")
    parser.add_argument("--verbose", default=True, type=bool, help="IT, ACCOM, BANK, WAIT")
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


def setup(args):
    """

    :param args:
    :return:
        cfg
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    if args.verbose == True:
        cfg.OUTPUT.VERBOSE = True
    cfg.merge_from_list(args.opts)

    return cfg


def train_test():
    """

    :return:
    """

    args = argument_parser().parse_args()
    cfg = setup(args)

    model = VISPEL(cfg, situ_decoding(args.situation), args.N)
    model.train_vispel()
    model.test_vispel()

    if cfg.OUTPUT.VERBOSE:
        print("Saved model !")

    save_model(model, args.model_name, cfg.OUTPUT.DIR)

if __name__ == '__main__':
    train_test()