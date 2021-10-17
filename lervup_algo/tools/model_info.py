"""
This module load the trained model and reproduce its results

Use:
    python model_info.py --model_name out/rcnn-bank_42.pkl
"""

import pickle
import _init_paths
import  argparse


def argument_parser():
    """
    Create a parser with some common arguments.

    Returns:
        argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required= True, help= "saved modeling name")

    return parser


def info():
    """

    :return:
    """
    args = argument_parser().parse_args()
    model = pickle.load(open(args.model_name,'rb'))
    model.cfg.OUTPUT.VERBOSE = True
    print('#-----------------------------------#')
    print('# MODEL CONFIGURATION')
    print("#-----------------------------------#")
    print(model.cfg)
    model.set_seeds()
    model.test_vispel()

    return model.test_result


if __name__ == '__main__':
    info()