"""
The module verifies the results from the best pre-trained models
according to different settings.

"""
from os.path import dirname, abspath
root = dirname(dirname(abspath(__file__)))
import sys
import glob
import tqdm
import random
import pickle
from pathlib import Path
import numpy as np


def set_seeds(seed_):
    random.seed(seed_)
    np.random.seed(seed_)


def run_lervup(setting):
    """Reproduce the results of pre-trained models by lervup

    :param model_path: str
        path to a folder containing pre-trained models.
    :param setting: str
        pre-training setting
            - fr: use only focal rating
            - no_fr: not use focal rating
            - half_objs: use only a half of visual concepts (objects) and fr
            - half_users: use only a half of users and fr
    """
    sys.path.insert(0,'./lervup_algo/lib')
    sys.path.insert(0,'./lervup_algo/vispel')
    sys.path.insert(0,'./lervup_algo/')

    if setting == 'fr':
        model_dir = root+'/pretrained_models/lervup/fr/'
    if setting == 'no_fr':
        model_dir = root+'/pretrained_models/lervup/no_fr/'
    if setting == 'half_objs':
        model_dir = root+'/pretrained_models/lervup/fr_rand_half_objects/'
    if setting == 'half_users':
        model_dir = root+'/pretrained_models/lervup/fr_rand_half_profiles/'

    model_paths = glob.glob(model_dir+'*')

    if setting != 'half_objs':
    # result of a pair of a detector (rcnn, mobi)
    # and a situation (it, bank, accom, wait)
        res_pair = {x.split('/')[-1].split('.')[0]:[] for x in model_paths}
    else:
        res_pair = {x.split('/')[-1].split('_seed_')[0]:[] for x in model_paths}

    for mpath in tqdm.tqdm(model_paths):
        
        if setting == 'half_objs':
            rmodel = pickle.load(open(mpath, 'rb'))
            rmodel.test_vispel(half_vis=True, load_half_vis= True) # use only a half of visual concepts.
            res_pair[mpath.split('/')[-1].split('_seed_')[0]].append(rmodel.test_result)
        else:
            rmodel = pickle.load(open(mpath, 'rb'))
            rmodel.test_vispel()
            res_pair[mpath.split('/')[-1].split('.')[0]].append(rmodel.test_result) 

    print('#-----------------------------------#')
    print('# TEST CORR')
    print("#-----------------------------------#")
    for pair, results in res_pair.items():
        print('*************')
        print('* '+pair)
        print('*************')
        print('test corr: '+"{:.4f}".format(np.mean(results)))
        print('\n')

    sys.path.remove('./lervup_algo/lib')
    sys.path.remove('./lervup_algo/vispel')
    sys.path.remove('./lervup_algo/')