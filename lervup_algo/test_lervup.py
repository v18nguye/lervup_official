"""
This module loads a trained model and reproduces its results

Use:
    python test_lervup.py --model_path ./mobinet_models/it_mobi.pkl --print 1
                         
"""
import sys
sys.path.append('./lib')
sys.path.append('./vispel')
import pickle
import  argparse


def argument_parser():
    """
    Create a parser with some common arguments.

    Returns:
        argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required= True, help= "saved modeling name")
    parser.add_argument("--print", default= '1', help= "print model info (1,0)")

    return parser


def test(model_path, print_ = True):
    """

    :return:
        val/test result
    """
    
    model = pickle.load(open(model_path,'rb'))
    model.cfg.OUTPUT.VERBOSE = True
    if print_:
        print('#-----------------------------------#')
        print('# MODEL CONFIGURATION')
        print("#-----------------------------------#")
        print(model.cfg)
        print('#-----------------------------------#')
        print('# TEST CORR')
        print("#-----------------------------------#")
        print("{:.4f}".format(model.test_result))

    return model.test_result


if __name__ == '__main__':
    args = argument_parser().parse_args()
    model_path = args.model_path
    print_ = False if args.print == '0' else True
    test(model_path, print_)