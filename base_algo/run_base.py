"""
Train and validate the ulmost base_algo method by
selecting a common optimal threshold for all objects (visual concepts).
"""

import os
from os.path import dirname, abspath
root = dirname(dirname(abspath(__file__)))
from .base.base import base
from .utils.data_loader import load_situs, load_gt_user_expo, load_train_test

def run_base(detector):
    """
    :param detector: str
        rcnn or mobinet
    """
    if not os.path.isdir(root+'/base_algo/base/out/'):
        os.mkdir(root+'/base_algo/base/out/')

    corr_type = 'pear_corr'
    gt_expo_path = root+'/dataset/lervup_data/gt_usr_exposure_v1.json'
    situation_file = root+'/dataset/lervup_data/visual_concepts/'

    if detector == 'rcnn':
        sav_path = root+'/base_algo/base/out/rcnn_common_optimal_thres_situs_v1.txt'
        train_test_path = root+'/dataset/lervup_data/train_test_split_rcnn_v1.json'
    
    elif detector == 'mobinet':
        sav_path = root+'/base_algo/base/out/mobi_common_optimal_thres_situs_v1.txt'
        train_test_path = root+'/dataset/lervup_data/train_test_split_mobinet_v1.json'

    else:
        raise ValueError('Detector' + detector + ' not deployed yet !')


    # Load crowdsourced user privacy exposures  in each situation
    gt_user_expo_situs = load_gt_user_expo(gt_expo_path)
    # Load train and test data
    train_user_subsets, val_user_ids = load_train_test(train_test_path)
    train_user_ids = train_user_subsets['100'] 
    # Read visual concept exposures in each situation
    visual_concept_scores = load_situs(situation_file)

    data =  {}
    data['train_user_ids'] = train_user_ids
    data['val_user_ids'] = val_user_ids
    data['visual_concept_scores'] = visual_concept_scores
    data['gt_user_exposure_scores'] = gt_user_expo_situs

    base(data, corr_type, sav_path, detector)