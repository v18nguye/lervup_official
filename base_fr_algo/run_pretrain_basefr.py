"""
Rerun the experiences 
by the baseline with focal rating.
                         
"""
from os.path import dirname, abspath
root = dirname(dirname(abspath(__file__)))
import glob
import sys
import pickle

def run_basefr():
    """

    """
    sys.path.insert(0,'./base_fr_algo/lib')
    model_paths = glob.glob(root+'/pretrained_models/base_fr/*/*')
    print('#-----------------------------------#')
    print('# TEST CORR')
    print("#-----------------------------------#")

    for mpath in model_paths:
        model = pickle.load(open(mpath,'rb'))

        print('*************')
        print('* '+mpath.split('/')[-1].split('.')[0])
        print('*************')
        print('test corr: '+"{:.4f}".format(model.test_result))
    sys.path.remove('./base_fr_algo/lib')