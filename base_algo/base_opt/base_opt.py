
import os
import sys
import json

from .lib import corr, tau_subset, search_optimal_thres

def base_opt(data, corr_type, sav_path, detector):
    """Run the baseline method with an optimal objectness-score 
    (object-detection-score) threshold per object (visual concept).
    
    :param data: dict
        dict containing data
            - ground-truth user exposures
            - visual concept scores
            - train/val user ids

    :param corr_type: str
        correlation type 

    :param sav_path: str
        save path

    """
    train_uids, val_uids = data['train_user_ids'], data['val_user_ids']
    obj_situs = data['visual_concept_scores']
    gt_uexpo_situs = data['gt_user_exposure_scores']
    
    load = True if os.path.isfile(sav_path) else False
    if not load:
        print('Calculating optimal objectness-score threholds per situation ....')
        opt_thres_situs = {}
        for situ, obj_dets in obj_situs.items():
            print(' ',situ)
            gt_uexpo = gt_uexpo_situs[situ]
            opt_thres_situs[situ] = search_optimal_thres(train_uids, gt_uexpo, obj_dets, corr_type)
        with open(sav_path, 'w') as fp:
            json.dump(opt_thres_situs, fp)
    else:
        print('Loading optimal objectness-score threholds per situation ...')   
        opt_thres_situs = json.load(open(sav_path))
    print('Done!')

    print('Estimating corr max, and opt detectors per situation ...')
    corr_max_situs = {}
    est_opt_det_situs = {} # estimate
    est_thres_situs = {}
    for situ, gt_uexpo in gt_uexpo_situs.items():
        print(' ',situ)
        corr_max, opt_dets, _, opt_thres = tau_subset(train_uids, gt_uexpo, opt_thres_situs[situ], corr_type)
        corr_max_situs[situ] = corr_max
        est_opt_det_situs[situ] = opt_dets
        est_thres_situs[situ] = opt_thres
    print('Done!')

    print('#-----------------------------------#')
    print('# ' +detector.upper()+ ' TEST CORR')
    print("#-----------------------------------#")
    for situ, gt_uexpo in gt_uexpo_situs.items():
        active_dets = est_opt_det_situs[situ]
        corr_situ = corr(val_uids, gt_uexpo, active_dets, corr_type, test_mode= True)
        print(situ+' corr: ', "{:.4f}".format(corr_situ))