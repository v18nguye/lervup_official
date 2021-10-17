"""
The module fine-tunes exposure predictor on the user selection,
 and loading methods.

run:
    python ft_official.py --config_file ../configs/rf_kmeans_ft_mobi_cv5.yaml --model_name best_bank.pkl --situation BANK
    python ft_official.py --config_file ../configs/rf_kmeans_ft_mobi_cv5.yaml --model_name best_accom.pkl --situation ACCOM
    python ft_official.py --config_file ../configs/rf_kmeans_ft_mobi_cv5.yaml --model_name best_it.pkl --situation IT
    python ft_official.py --config_file ../configs/rf_kmeans_ft_mobi_cv5.yaml --model_name best_wait.pkl --situation WAIT

"""
import _init_paths
import os
import tqdm
import copy
import numpy as np
import argparse
import pickle
from situ.acronym import situ_decoding
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
    parser.add_argument("--fe", required=True, help="1 to turn on the focal exposure")
    parser.add_argument("--situation", required=True, help="IT, ACCOM, BANK, WAIT")
    parser.add_argument("--N", required=True, help="Number of training profiles: "
                                                   "-1: ALL user profiles"
                                                   "N: N profiles (N < 400)")
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


def set_up(args):
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

    :return:
    """
    trained_models = []
    test_corrs = []

    args = argument_parser().parse_args()
    cfg = set_up(args)

    DETECTOR_LOADs = [True, False]
    F_TOPs = [0, 0.2, 0.3]
    FILTs = [0.1]

    if args.fe == 1:
        Ks = [10, 15, 20, 25]
        GAMMAs = [0, 1, 2, 3, 4]
    else:
        Ks = [10]
        GAMMAs = [0]

    for load in tqdm.tqdm(DETECTOR_LOADs):
        cfg.DETECTOR.LOAD = load

        for f_top in F_TOPs:
            cfg.SOLVER.F_TOP = f_top

            for gamma in GAMMAs:
                cfg.FE.GAMMA = gamma

                for filt in FILTs:
                    cfg.SOLVER.FILT = filt

                    for k in Ks:
                        cfg.FE.K = k

                        model = VISPEL(cfg, situ_decoding(args.situation), args.N)

                        model.train_vispel()
                        trained_models.append(copy.deepcopy(model))

                        model.test_vispel()
                        test_corrs.append(model.test_result)

                        del model

    if cfg.OUTPUT.VERBOSE:
        print("Save modeling !!!")

    max_corr_index = np.argmax(test_corrs)
    best_model = trained_models[max_corr_index]
    save_model(best_model, args.model_name, cfg.OUTPUT.DIR)
    print(test_corrs)
    print('Best Model Result: ', test_corrs[max_corr_index])


if __name__ == '__main__':
    main()
