"""
Train and validate the baseline method with an optimal threshold for individual concept
and only an active set of detectors for each siutation.

"""

import os
from os.path import dirname, abspath
root = dirname(dirname(abspath(__file__)))
from .base_opt.base_opt import base_opt
from .utils.data_loader import load_situs, load_gt_user_expo, load_train_val_test

def run_base_opt():
    """

    """
    if not os.path.isdir(root+'/base_algo/base_opt/out/'):
        os.mkdir(root+'/base_algo/base_opt/out/')

    corr_type = 'pear_corr'

    gt_expo_path = root+'/dataset/iclef_data/gt_trainvaltest_uexposure.json'
    situation_file = root+'/dataset/iclef_data/visual_concepts/'
    sav_path = root+'/base_algo/base_opt/out/iclef_optimal_thres_situs.txt'
    train_test_path = root+'/dataset/iclef_data/train-val-test_detection_split_iclef.json'


    # Load crowdsourced user privacy exposures  in each situation
    gt_user_expo_situs = load_gt_user_expo(gt_expo_path)
    # Load train and test data
    train_user_ids, val_user_ids, test_user_ids = load_train_val_test(train_test_path)
    # Read visual concept exposures in each situation
    visual_concept_scores = load_situs(situation_file)

    data =  {}
    data['train_user_ids'] = train_user_ids
    data['val_user_ids'] = val_user_ids
    data['test_user_ids'] = test_user_ids
    data['visual_concept_scores'] = visual_concept_scores
    data['gt_user_exposure_scores'] = gt_user_expo_situs

    base_opt(data, corr_type, sav_path)