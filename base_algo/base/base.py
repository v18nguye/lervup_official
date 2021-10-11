import os
import json
from .lib import corr, search_common_thres

def base(data, corr_type, sav_path, detector):
    """Run the baseline method with a common threshold for
    all objects (visual concepts)
    
    """
    train_uids, val_uids = data['train_user_ids'], data['val_user_ids']
    obj_situs = data['visual_concept_scores']
    gt_uexpo_situs = data['gt_user_exposure_scores']

    load = True if os.path.isfile(sav_path) else False
    if not load:
        print('Calculating an common optimal objectness-score threhold per situation ....')
        opt_thres_situs = {}
        for situ, obj_dets in obj_situs.items():
            print(' ',situ)
            gt_uexpo = gt_uexpo_situs[situ]
            opt_thres_situs[situ] = search_common_thres(train_uids, gt_uexpo, obj_dets, corr_type)
        with open(sav_path, 'w') as fp:
            json.dump(opt_thres_situs, fp)
    else:
        print('Loading optimal objectness-score threholds per situation ...')   
        opt_thres_situs = json.load(open(sav_path))
    print('Done!')

    print('#-----------------------------------#')
    print('# ' +detector.upper()+ ' TEST CORR')
    print("#-----------------------------------#")
    for situ, gt_uexpo in gt_uexpo_situs.items():
        opt_thres = opt_thres_situs[situ]
        obj_dets = obj_situs[situ]
        detectors = {}
        for det, score in obj_dets.items():
            detectors[det] = (opt_thres, score)
        corr_situ = corr(val_uids, gt_uexpo, detectors, corr_type)
        print(situ+' corr: ', "{:.4f}".format(corr_situ))