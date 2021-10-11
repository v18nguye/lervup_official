import math
import numpy as np
import tqdm
from scipy.stats import kendalltau, pearsonr


############################################
#   EXPOSURE INFO TOOLS
############################################

def photo_exposure(photo, detectors, test_mode = False):
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
            if not test_mode:
                if max(obj_scores) >= detectors[class_][0]:
                    active_state = True
                    photo_score += max(obj_scores)*detectors[class_][1]
            else:
                active_state = True
                photo_score += max(obj_scores) * detectors[class_][1]

    return photo_score, active_state


def user_expo(photos, detectors, test_mode):
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

        photo_score, active_state = photo_exposure(detected_objects, detectors, test_mode)
        user_score += photo_score

        if active_state:
            cardinality += 1

    if cardinality != 0:
        user_score = user_score/cardinality

    return user_score


def user_expo_situ(users, detectors, test_mode):
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
        community_expo[user] = user_expo(photos, detectors, test_mode)

    return community_expo


############################################
#   CORRELATION TOOLS
############################################

def corr(data, gt_user_expo, detectors, corr_type, test_mode= False):
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
    user_scores = user_expo_situ(data, detectors, test_mode)
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


def search_thres(train_data, gt_user_expo, detector_score, corr_type):
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
        tau = corr(train_data, gt_user_expo, detector, corr_type)
        if math.isnan(tau):
            tau = 0
        tau_list.append(tau)

    tau_max = max(tau_list)
    threshold_max = threshold_list[np.argmax(tau_list)]

    return tau_max, threshold_max


def search_optimal_thres(train_data, gt_user_expo, detectors, corr_type):
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

    for detector, score in tqdm.tqdm(detectors.items()):
        detector_score = [detector,score]
        tau_max, threshold_max = search_thres(train_data, gt_user_expo, detector_score, corr_type)
        max_tau_detectors[detector] = (tau_max, threshold_max, score)

    return max_tau_detectors


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


def tau_subset(users, gt_user_expo, detectors, corr_type):
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
    
    for tau_fix in tqdm.tqdm(tau_fixes):
        detector_subset = select_subset(detectors, tau_fix)

        tau_est = corr(users, gt_user_expo, detector_subset, corr_type)

        if math.isnan(tau_est):
            tau_est = 0
        list_tau_estimate.append(tau_est)
        list_opt_detectors.append(detector_subset)

    tau_max = max(list_tau_estimate)
    opt_detectors = list_opt_detectors[np.argmax(list_tau_estimate)]
    threshold = tau_fixes[np.argmax(list_tau_estimate)]

    return tau_max, opt_detectors, list_tau_estimate, threshold