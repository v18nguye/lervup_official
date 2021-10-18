import math
import numpy as np
from .correlation import corr


def select_subset(detectors, tau_fix):
    """
    Select a subset whose detector taus are greater than tau_fix

    :param: detectors: dict
        {detector1: (tau_max1, threshold1, score1), ...}

    :param: tau_fix: float

    :return:
        tau_detectors: dict
            {detector1: (threshold1, score1), ...}

    """
    detector_subset = {}

    for detector, tau_thres_score in detectors.items():
        if tau_thres_score[0] >= tau_fix or abs(tau_thres_score[2]) >= 1:
            detector_subset[detector] = [tau_thres_score[1], tau_thres_score[2]]
    
    return detector_subset


def tau_subset(users, gt_user_expo, detectors, corr_type, cfg):
    """
    Estimate the best correlation score for a subset tau_detectors

    :param: users
        users in a situation and its photos
            {user1: {photo1: {class1: [obj1, ...], ...}}, ...}, ...}
            
    :param gt_user_expo: dict
        user expo in a given situation
            {user1: avg_score, ...}
    
    :param: detectors: dict
        {detector1: (tau_max1, threshold1, score1), ...}
    
    :param corr_type: string

    :return:

    """
    list_opt_detectors = []
    list_tau_estimate = []
    tau_fixes = list(np.linspace(-1, 1, 201))
    tau_fixes = [float("{:.2f}".format(tau)) for tau in tau_fixes]
    
    for tau_fix in tau_fixes:
        detector_subset = select_subset(detectors, tau_fix)

        tau_est = corr(users, gt_user_expo, detector_subset, corr_type, cfg)

        if math.isnan(tau_est):
            tau_est = 0
        list_tau_estimate.append(tau_est)
        list_opt_detectors.append(detector_subset)

    tau_max = max(list_tau_estimate)
    opt_detectors = list_opt_detectors[np.argmax(list_tau_estimate)]
    threshold = tau_fixes[np.argmax(list_tau_estimate)]

    return tau_max, opt_detectors, list_tau_estimate, threshold


def tau_max_cross_val(users, gt_user_expo, detectors, corr_type, cfg, k_fold = 5):
    """Search tau max by cross validation

    :param: users
        users in a situation and its photos
            {user1: {photo1: {class1: [obj1, ...], ...}}, ...}, ...}

    :param gt_user_expo: dict
        user expo in a given situation
            {user1: avg_score, ...}

    :param: detectors: dict
        {detector1: (tau_max1, threshold1, score1), ...}

    :param: detectors: dict
        {detector1: (tau_max1, threshold1, score1), ...}

    :param corr_type:

    :param k_fold: number of folds

    :return:

    """
    test_fold_size = int(len(list(users.keys()))/k_fold)

    threshold_dict = {}

    for index in range(k_fold):

        start = index*test_fold_size
        end = (index + 1)*test_fold_size
        count = 0
        train_fold = {}
        val_fold = {}

        for user, photos in users.items():

            if count >= start and count < end:
                val_fold[user] = photos
            else:
                train_fold[user] = photos

            count += 1

        tau_max, opt_detectors, _, threshold = tau_subset(train_fold, gt_user_expo, detectors, corr_type, cfg)

        if str(threshold) not in threshold_dict:
            threshold_dict[str(threshold)] = {'score_val': [], 'opt_detectors': opt_detectors}

        score_val = corr(val_fold, gt_user_expo, opt_detectors, corr_type, cfg)

        threshold_dict[str(threshold)]['score_val'].append(score_val)

    for threshold, items in threshold_dict.items():
        threshold_dict[threshold]['score_val'] = sum(items['score_val'])/len(items['score_val'])

    score_val_max = -1

    for threshold, items in threshold_dict.items():
        if items['score_val'] >= score_val_max:
            opt_threshold = float(threshold)
            score_val_max = items['score_val']
            opt_detectors = items['opt_detectors']

    return score_val_max, opt_threshold, opt_detectors