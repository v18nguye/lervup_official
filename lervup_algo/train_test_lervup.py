"""
The module trains user's exposure predictors using the proposed lervup algo.

run with default settings:

    python train_test_lervup.py --config_file ./configs/mobi_rf_kmeans.yaml --model_name it_mobi.pkl --situation IT --N -1 --fr 0 --opts OUTPUT.DIR ./lervup/no_fr/
    python train_test_lervup.py --config_file ./configs/mobi_rf_kmeans.yaml --model_name accom_mobi.pkl --situation ACCOM --N -1 --fr 0 --opts OUTPUT.DIR ./lervup/no_fr/
    python train_test_lervup.py --config_file ./configs/mobi_rf_kmeans.yaml --model_name bank_mobi.pkl --situation BANK --N -1 --fr 0 --opts OUTPUT.DIR ./lervup/no_fr/
    python train_test_lervup.py --config_file ./configs/mobi_rf_kmeans.yaml --model_name wait_mobi.pkl --situation WAIT --N -1 --fr 0 --opts OUTPUT.DIR ./lervup/no_fr/

    python train_test_lervup.py --config_file ./configs/rcnn_rf_kmeans.yaml --model_name it_rcnn.pkl --situation IT --N -1 --fr 0 --opts OUTPUT.DIR ./lervup/no_fr/
    python train_test_lervup.py --config_file ./configs/rcnn_rf_kmeans.yaml --model_name accom_rcnn.pkl --situation ACCOM --N -1 --fr 0 --opts OUTPUT.DIR ./lervup/no_fr/
    python train_test_lervup.py --config_file ./configs/rcnn_rf_kmeans.yaml --model_name bank_rcnn.pkl --situation BANK --N -1 --fr 0 --opts OUTPUT.DIR ./lervup/no_fr/
    python train_test_lervup.py --config_file ./configs/rcnn_rf_kmeans.yaml --model_name wait_rcnn.pkl --situation WAIT --N -1 --fr 0 --opts OUTPUT.DIR ./lervup/no_fr/


run with more customized settings:

    python3 train_test_lervup.py --config_file ./configs/rf_kmeans_ft_mobi_cv5.yaml --model_name it_mobi.pkl --situation IT --N -1 --opts FE.MODE OBJECT  FE.K 10  FE.GAMMA 2  DETECTOR.LOAD True SOLVER.PFT ORG OUTPUT.DIR ./mobinet_models/

The VISPEL module stands for VISual Photo Exposure Learning.
"""
import os
import sys
import time
sys.path.append('./lib/')
sys.path.append('./vispel/')
import argparse
import pickle
from tools.ftune_hprams import ftune_model_hyp
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
    parser.add_argument("--fr", required=True, type = int, help="if use Focal Rating ( > 0 is True)")
    parser.add_argument("--N", default=-1, type = int, help="Number of training profiles: "
                                                   "-1: ALL user profiles"
                                                   "N: N profiles [-1, 175]")
    parser.add_argument("--verbose", default=0)
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
    if int(args.verbose) == 0:
        cfg.OUTPUT.VERBOSE = False
    else:
        cfg.OUTPUT.VERBOSE = True
    cfg.merge_from_list(args.opts)

    return cfg


def train_val():
    """

    :return:
    """

    args = argument_parser().parse_args()
    cfg = setup(args)
    print(args.model_name)
    start = time.time()
    model = VISPEL(situ_decoding(args.situation), args.N)
    trained_model, val_score = ftune_model_hyp(model, cfg, args)

    print('total time: ', time.time() - start)

    if cfg.OUTPUT.VERBOSE:
        print("Saved model !")

    save_model(trained_model, args.model_name, cfg.OUTPUT.DIR)
    print('\n\n\n')

if __name__ == '__main__':
    train_val()