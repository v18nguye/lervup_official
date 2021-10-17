"""
The module verifies the results from the best pre-trained models
according to different settings.

"""
import sys
# sys.path.append('./lib/')
# sys.path.append('./vispel/')
import glob
import pickle
import random
import numpy as np
from pathlib import Path
import warnings
import _init_paths
warnings.filterwarnings("ignore")

# def argument_parser():
#     """
#     Create a parser with some common arguments.

#     Returns:
#         argparse.ArgumentParser
#     """
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--setting", required=True, choices=['fr', 'no_fr' ,'half_objs', 'half_users'], help="pretrained-model setting")

#     return parser

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
    root = '/home/nguyen/Documents/intern20/lervup_offical'
    if setting == 'fr':
        model_dir = '/pretrained_models/lervup/fr/'
    if setting == 'no_fr':
        model_dir = '/pretrained_models/lervup/no_fr/'
    if setting == 'half_objs':
        model_dir = '/pretrained_models/lervup/fr_rand_half_objects/'
    if setting == 'half_users':
        model_dir = '/pretrained_models/lervup/fr_rand_half_profiles/'

    model_paths = glob.glob(root + model_dir+'*')

    if setting != 'half_objs':
    # result of a pair of a detector (rcnn, mobi)
    # and situation (it, bank, accom, wait)
        res_pair = {x.split('/')[-1].split('.')[0]:[] for x in model_paths}
    else:
        res_pair = {x.split('/')[-1].split('_seed_')[0]:[] for x in model_paths}

    for seed in [10, 100, 1000, 10000, 100000]:
        # set_seeds(seed)
        for mpath in model_paths:
            if int(mpath.split('/')[-1].split('_seed_')[-1].split('.')[0]) == seed:
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
    print(res_pair)
    for pair, results in res_pair.items():
        print('*************')
        print('* '+pair)
        print('*************')
        print('test corr: '+"{:.4f}".format(np.mean(results)))
        print('\n')


if __name__ == '__main__':
    run_lervup(setting='half_objs')