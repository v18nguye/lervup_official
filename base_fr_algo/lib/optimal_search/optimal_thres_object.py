import math
import numpy as np
from .correlation import corr

def search_thres(train_data, gt_user_expo, detector_score, corr_type, cfg):
    """

    :param train_data:
    :param gt_user_expo:
    :param detector_score: list
            [detector, object_score]
                + detector: the type of object need to searched for
                + object_score: crowd-sourcing object score
    :param corr_type:

    :return:
        the best threshold for the given object

    """
    threshold_list = [float("{:.2f}".format(0.01*i)) for i in range(101)]
    tau_list = []

    for threshold in threshold_list:
        detector = {detector_score[0]: (threshold, detector_score[1])}
        tau = corr(train_data, gt_user_expo, detector, corr_type, cfg)
        if math.isnan(tau):
            tau = 0
        tau_list.append(tau)

    tau_max = max(tau_list)
    threshold_max = threshold_list[np.argmax(tau_list)]

    return tau_max, threshold_max


def search_optimal_thres(train_data, gt_user_expo, detectors, corr_type, cfg):
    """Search optimal object thresholds
    for all type of object within a given correlation type 
    
    :param train_data: dict
        users and images in training data
            {user1: {photo1: {class1: [obj1, ...], ...}}, ...}, ...}

    :param gt_user_expo: dict
        user expo in a given situation
            {user1: avg_score, ...}

    :param detectors: dict
        all detectors in a given situation
            {detector1: score1, detector2: score2, ...}

    :param corr_type: string
        correlation type:
            + pear_corr
            + kendall_corr
    
    :return
        max_tau_detectors: dict
            {object1: (tau_max_1, threshold1, score1), ...}
    """
    max_tau_detectors = {}

    for detector, score in detectors.items():
        detector_score = [detector,score]
        tau_max, threshold_max = search_thres(train_data, gt_user_expo, detector_score, corr_type, cfg)
        max_tau_detectors[detector] = (tau_max, threshold_max, score)

    return max_tau_detectors