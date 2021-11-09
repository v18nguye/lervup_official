"""
This module loads a trained model and reproduces its results

Use:
    python test_lervup.py --model_dir ./lervup/fr/
                         
"""
import sys
# sys.path.append('./lib')
# sys.path.append('./vispel')
import pickle
import glob
import  argparse
from loader.data_loader import data_loader


def argument_parser():
    """
    Create a parser with some common arguments.

    Returns:
        argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default= './lervup/fr/', help= "saved modeling name")

    return parser


def test_lervup(model_dir):
    """

    :return:
        val/test result
    """
    print('#-----------------------------------#')
    print('# VAL and TEST CORR')
    print("#-----------------------------------#")
    model_paths = glob.glob(model_dir+'*')
    for model_path in model_paths:
        model = pickle.load(open(model_path,'rb'))
        # print(model.cfg)
        X_train, X_val, X_test, \
                    gt_expos, \
                    detectors, opt_threds = data_loader(model.cfg, model.situ_name)
        print(model_path)
        print("val corr: "+"{:.2f}".format(model.test_vispel(X_val, gt_expos, detectors, opt_threds)))
        print("test corr: "+"{:.2f}".format(model.test_vispel(X_test, gt_expos, detectors, opt_threds)))
        print('\n\n')
        print('***************')


if __name__ == '__main__':
    args = argument_parser().parse_args()
    model_dir = args.model_dir
    test_lervup(model_dir)