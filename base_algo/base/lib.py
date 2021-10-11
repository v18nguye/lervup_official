import math
import numpy as np
import tqdm
from scipy.stats import kendalltau, pearsonr


############################################
#   EXPOSURE INFO TOOLS
############################################

def photo_exposure(photo, detectors):
    """

    :param photo: dict
        {class1: [obj1, ...], ...}

    :param detectors: dict
        {detector1: (threshold, object_score), ...}

    :return:
        activate : boolean
            does the photo have at least one detector

        photo_score: float
    """

    active_state = False

    photo_score = 0
    for class_, obj_scores in photo.items():
        if class_ in detectors:
            if max(obj_scores) >= detectors[class_][0]:
                active_state = True
                photo_score += max(obj_scores)*detectors[class_][1]

    return photo_score, active_state


def user_expo(photos, detectors):
    """Estimate user exposure

    :param photos: dict
        user photos and its detected objects
            {photo1: {class1: [obj1, ...], ...}}, ...}

    :param detectors: dict
         {detector: (threshold, object_score),...} for not inference_mode
        {detector1: object_score, ...} for inference_mode

    :return:
        user_score: float
    """

    user_score = 0
    cardinality = 0
    for photo, detected_objects in photos.items():

        photo_score, active_state = photo_exposure(detected_objects, detectors)
        user_score += photo_score

        if active_state:
            cardinality += 1

    if cardinality != 0:
        user_score = user_score/cardinality

    return user_score


def user_expo_situ(users, detectors):
    """

    :param users:
        users in a situation and its photos
            {user1: {photo1: {class1: [obj1, ...], ...}}, ...}, ...}

    :param detectors: dict
        {detector1: (threshold, object_score), ...}

    :return:
        community_expo: dict
            {user1: score, ...}

    """
    community_expo = {}
    for user, photos in users.items():
        community_expo[user] = user_expo(photos, detectors)

    return community_expo


############################################
#   CORRELATION TOOLS
############################################

def corr(data, gt_user_expo, detectors, corr_type):
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
    user_scores = user_expo_situ(data, detectors)
    automatic_eval = []
    manual_eval = []

    for user, score in user_scores.items():
        automatic_eval.append(score)
        manual_eval.append(gt_user_expo[user])

    automatic_eval = np.asarray(automatic_eval)
    manual_eval = np.asarray(manual_eval)

    if corr_type == 'pear_corr':
        tau, _ = pearsonr(automatic_eval,manual_eval)
    elif corr_type == 'kendall_corr':
        tau, _ = kendalltau(automatic_eval, manual_eval)

    return tau

def search_common_thres(train_data, gt_user_expo, detector_score, corr_type):
    """Search an optimal threshold for all objects (visual concepts)

    :param train_data:
    :param gt_user_expo:
    :param detector_score: dict
        object detector and its score
    :param corr_type:

    :return:
        the optimal threshold for the given objects

    """
    threshold_list = [float("{:.2f}".format(0.01*i)) for i in range(101)]
    corr_list = []

    for threshold in tqdm.tqdm(threshold_list):
        detectors = {}
        for det, score in detector_score.items():
            detectors[det] = (threshold, score)
        tau = corr(train_data, gt_user_expo, detectors, corr_type)
        if math.isnan(tau):
            tau = 0
        corr_list.append(tau)

    opt_threshold = threshold_list[np.argmax(corr_list)]

    return opt_threshold