"""
This module loads all trained models in a directory and reproduces its results

Use:
    python test_trained_models.py --model_dir ./base_fr/ --print 1
                         
"""
import sys
sys.path.append('./lib')
import pickle
import glob
import  argparse


def argument_parser():
    """
    Create a parser with some common arguments.

    Returns:
        argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required= True, help= "saved modeling name")
    parser.add_argument("--print", default= '1', help= "print model info (1,0)")

    return parser


def test(model_dir, print_ = True):
    """

    :return:
        val/test result
    """
    model_paths = [x for x in glob.glob(model_dir+'*/*') if '.txt' not in x]

    print('#-----------------------------------#')
    print('# VAL and TEST CORR')
    print("#-----------------------------------#")

    for model_path in model_paths:
        model = pickle.load(open(model_path,'rb'))
        model.cfg.OUTPUT.VERBOSE = True
        print(model_path)
        print("val corr: "+"{:.2f}".format(model.test(model.x_val)))
        print("corr: "+"{:.2f}".format(model.test(model.x_test)))
        print('\n\n')


if __name__ == '__main__':
    args = argument_parser().parse_args()
    model_dir = args.model_dir
    print_ = False if args.print == '0' else True
    test(model_dir, print_)