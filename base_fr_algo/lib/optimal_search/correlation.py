import numpy as np
from scipy.stats import kendalltau, pearsonr
from exposure.user import user_expo_situ

def corr(data, gt_user_expo, detectors, corr_type, cfg, test_mode= False):
    """Calculate correlation score for a threshold

    :param data: dict
        users and images in training data
            {user1: {photo1: {class1: [obj1, ...], ...}}, ...}, ...}

    :param gt_user_expo: dict
        user expo in a given situation
            {user1: avg_score, ...}

    :param detectors: dict
        the type of object need to searched for
            {detector: (thres, object_score), ...} for not inference_mode
                + thres: a given considered threshold
                + object_score: crowd-sourcing object score
            {detector1: object_score, ...} for inference_mode

    :param corr_type: string
        correlation type:
            + pear_corr
            + kendall_corr

    :return:
        tau: float
            correlation
    """
    user_scores = user_expo_situ(data, detectors, cfg, test_mode)
    automatic_eval = []
    manual_eval = []

    for user, score in user_scores.items():
        automatic_eval.append(score)
        manual_eval.append(gt_user_expo[user])

    automatic_eval = np.asarray(automatic_eval)
    manual_eval = np.asarray(manual_eval)


    if corr_type == 'PEARSON':
        tau, _ = pearsonr(automatic_eval,manual_eval)
    elif corr_type == 'KENDALL':
        tau, _ = kendalltau(automatic_eval, manual_eval)

    return tau