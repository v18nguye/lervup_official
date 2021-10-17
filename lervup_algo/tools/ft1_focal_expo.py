"""
This module fine-tunes models
on the focal exposure's gamma parameters

Use:
    python ft1_focal_expo.py --pre_model out/rcnn-bank_ft0.pkl --model_name rcnn-bank_ft1.pkl > out/rcnn-bank_ft1.txt
"""


import _init_paths
import os
import tqdm
import copy
import pickle
import  argparse
import numpy as np


def default_argument_parser():
    """
    Create a parser with some common arguments.

    Returns:
        argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre_model", required= True, help= "pretrained model")
    parser.add_argument("--model_name", required= True, help= "saved modeling name")

    return parser


def save_model(model, filename, out_dir):
    root = os.getcwd()
    out_dir_path = os.path.join(root, out_dir)
    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path)

    out_file_path = os.path.join(out_dir_path, filename)

    with open(out_file_path, 'wb') as output:
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)


def main():
    """

    :return:
    """
    args = default_argument_parser().parse_args()
    model = pickle.load(open(args.pre_model, 'rb'))

    trained_models = []
    test_corrs = []

    model.cfg.FINE_TUNING.STATUS = True
    model.cfg.FINE_TUNING.CV = 8
    model.cfg.REGRESSOR.RF.BOOTSTRAP = [True, False]
    model.cfg.REGRESSOR.RF.MAX_DEPTH = [5,7,9]
    model.cfg.REGRESSOR.RF.MAX_FEATURES = ['auto']
    model.cfg.REGRESSOR.RF.N_ESTIMATORS = [80, 120, 160]
    model.cfg.REGRESSOR.RF.MIN_SAMPLES_LEAF = [2,3]
    model.cfg.REGRESSOR.RF.MIN_SAMPLES_SPLIT= [2,3]

    Ks = [10,15,20]
    GAMMAs = [1, 2, 3]

    for gamma in tqdm.tqdm(GAMMAs):
        model.cfg.SOLVER.GAMMA = gamma
        print('#----------------------#')
        print('GAMMA: ',gamma)
        for k in Ks:
            model.cfg.SOLVER.K = k

            model.set_seeds()
            model.train_vispel()
            model.test_vispel()

            trained_models.append(copy.deepcopy(model))
            test_corrs.append(model.test_result)
            print(model.test_result)

    if model.cfg.OUTPUT.VERBOSE:
        print("Saving model !")

    max_corr_index = np.argmax(test_corrs)
    best_model = trained_models[max_corr_index]
    save_model(best_model, args.model_name, model.cfg.OUTPUT.DIR)
    print(test_corrs)
    print('Best Model Result: ',test_corrs[max_corr_index])

if __name__ == '__main__':
    main()