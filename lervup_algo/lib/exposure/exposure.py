import numpy as np
from exposure.focal_exposure import focal_exposure as FE


def feature_transform(f_expo_pos, f_expo_neg, f_dens, transform):
    """
    Apply feature transform on photo features scaled by focal exposure

    :param f_expo_pos:
    :param f_expo_neg:
    :param f_dens:
    :param transform: transforming method
    :return:
        transformed features

    """
    if transform == 'ORG':
        return [f_expo_pos, f_expo_neg, f_dens]

    if transform == 'VOTE':
        if f_expo_pos > abs(f_expo_neg):
            f_expo = f_expo_pos
        else:
            f_expo = f_expo_neg
        return [f_expo, f_dens]


def photo_expo(photo, f_top, detectors, opt_threshs, load_detectors, cfg):
    """Estimate photo exposure

    Parameters
    ----------
    photo : dict
        objects in photo associating its detection confidence
            {class1: [obj1, obj2,...], ... }

    f_top : float [0,1)
        object detection confidence scores > f_top for a given
        concept.

    load_detectors : boolean
        load active detectors pre-computed by the privacy
        base-line method

    detectors : dict
        all active detectors in a given situation and its scores
            {detector1: score, ...}

    opt_threds:
            optimal threshold for each object. Precomputed by the base line privacy method.

    Returns
    -------
        expo_obj : tuple
            photo exposure features and its objectness sum
                {exp +, expo -, objness}
    """

    expo_pos = []  # positive exposure
    expo_neg = []  # negative exposure
    objectness = []

    extrem_pos_concepts = [] 
    extrem_neg_concepts = []
    neutral_pos_concepts = []
    neutral_neg_concepts = []
    scale_flag = False

    # Statistics on extreme/ neutral exposure concepts
    for object_, scores in photo.items():
        
        obj_score = 0
        if object_ in detectors:
            # TAU_e = 1 (80% percentile of all visual concept scores)
            if detectors[object_] > cfg.FE.TAU_e: 
                extrem_pos_concepts.append(object_)
            elif detectors[object_] < -cfg.FE.TAU_e:
                extrem_neg_concepts.append(object_)

            if 0 <= detectors[object_] <= cfg.FE.TAU_e:
                neutral_pos_concepts.append(object_)
            elif -cfg.FE.TAU_e <= detectors[object_] < 0:
                neutral_neg_concepts.append(object_)

            # Estimate object-ness of the photo, which is equal
            # to mean of valid objectness scores.

            if not load_detectors:
                valid_obj = [score for score in scores if score >= f_top]
                if sum(valid_obj) > 0:
                    obj_score += sum(valid_obj) / len(valid_obj)
            else:
                valid_obj = [score for score in scores if score >= opt_threshs[object_]]
                if sum(valid_obj) > 0:
                    obj_score += sum(valid_obj) / len(valid_obj)
            objectness.append(obj_score)
    objectness = sum(objectness) / len(objectness) if sum(objectness) != 0 else 0

    # Focal Rating modes
    if cfg.FE.MODE == 'IMAGE':
        if len(neutral_pos_concepts) != 0:
            ratio = len(extrem_pos_concepts) / (len(neutral_pos_concepts) + len(neutral_neg_concepts) + len(extrem_pos_concepts))
            scale_pos_flag = True if 0 < ratio < cfg.FE.TAU_o else 0 # TAU_o = 1/3
        else:
            scale_pos_flag = False

        if len(neutral_neg_concepts) != 0:
            ratio = len(extrem_neg_concepts) / (len(neutral_pos_concepts) + len(neutral_neg_concepts) + len(extrem_neg_concepts))
            scale_neg_flag = True if 0 < ratio < cfg.FE.TAU_o else False
        else:
            scale_neg_flag = False
    elif cfg.FE.MODE == 'OBJECT':
        scale_pos_flag = True
        scale_neg_flag = True

    # Compute photo exposure features.
    for object_, scores in photo.items():
        if object_ in detectors:
            
            scaled_expo = 0
            if detectors[object_] >= 0:
                if not load_detectors:
                    if scale_pos_flag and max(scores) > f_top:
                        scale_flag = True
                        scaled_expo = FE(detectors[object_], cfg.FE.GAMMA, cfg.FE.K)
                    elif max(scores) > f_top:
                        scaled_expo = detectors[object_]
                else:
                    if scale_pos_flag and max(scores) > opt_threshs[object_]:
                        scale_flag = True
                        scaled_expo = FE(detectors[object_], cfg.FE.GAMMA, cfg.FE.K)
                    elif max(scores) > opt_threshs[object_]:
                        scaled_expo = detectors[object_]
                expo_pos.append(scaled_expo)

            if detectors[object_] < 0:
                if not load_detectors:
                    if scale_neg_flag and max(scores) > f_top:
                        scale_flag = True
                        scaled_expo = FE(detectors[object_], cfg.FE.GAMMA, cfg.FE.K)
                    elif max(scores) > f_top:
                        scaled_expo = detectors[object_]
                else:
                    if scale_neg_flag and max(scores) > opt_threshs[object_]:
                        scale_flag = True
                        scaled_expo = FE(detectors[object_], cfg.FE.GAMMA, cfg.FE.K)
                    elif max(scores) > opt_threshs[object_]:
                        scaled_expo = detectors[object_]
                expo_neg.append(scaled_expo)

    expo_pos = sum(expo_pos) / len(expo_pos) if sum(expo_pos) != 0 else 0
    expo_neg = sum(expo_neg) / len(expo_neg) if sum(expo_neg) != 0 else 0
    expo_obj = (expo_pos, expo_neg, objectness)

    return expo_obj, scale_flag


def user_expo(user_photos, f_top, detectors, opt_threshs, load_detectors, cfg):
    """Estimate user's photo exposure features.

    Parameters
    ----------
        user_photos : dict
            user photos associated with object confidence scores
                {photo1: {class1: [obj1, ...], ...},...}

        load_detectors : boolean
            if load active detectors pre-computed by the privacy
            base-line method

        opt_threshs: dict
            active detectors pre-computed by the privacy
            base-line method 

        f_top : float [0,1)
            object detection confidence scores > f_top for a given
            concept.

        detectors : dict
            all visual concept detectors for a given situation

        filter : boolean
            filtering neutral photos with a threshold
        
    Returns
    -------
        expo : dict
            user photo feature exposure
                {photo1: [transformed features],...}

    """
    expo = {}
    count_rescaled_imgs = []

    for photo in user_photos:
        (pos_expo, neg_expo, objectness), scale_flag = photo_expo(user_photos[photo], f_top, detectors, opt_threshs,
                                                                    load_detectors, cfg)

        # PHOTO EXPOUSURE FEATURES
        f_expo_pos = pos_expo
        f_expo_neg = neg_expo
        f_dens = objectness

        if scale_flag:
            count_rescaled_imgs.append(photo)

        # Apply feature transform
        if cfg.SOLVER.FILTERING: # filtering neural images
            if abs(f_expo_pos) + abs(f_expo_neg) >= cfg.SOLVER.FILT_THRESHOLD:    
                expo[photo] = feature_transform(f_expo_pos, f_expo_neg, f_dens, cfg.SOLVER.FEATURE_TYPE)
        else:
            expo[photo] = feature_transform(f_expo_pos, f_expo_neg, f_dens, cfg.SOLVER.FEATURE_TYPE)

    return expo, len(count_rescaled_imgs), len(user_photos)


def community_expo(users, f_top, detectors, opt_threshs, load_detectors, cfg):
    """Estimate photo exposure features for  community's users.

    Parameters
    ----------
        users : dict
            users and their photos
                {user1: {photo1: {class1: [obj1, ...], ...}, ...}, ...}
        
        f_top : float [0,1)
            object detection confidence scores > f_top for a given
            concept.

        load_detectors : boolean
            if load active detectors pre-computed by the privacy
            base-line method

        opt_threshs: dict
            active detectors pre-computed by the privacy
            base-line method 

        detectors : dict
            all visual concept detectors for a given situation

        opt_threshs : dict
            optimal active detector thresholds
                {detector1: thresh1, ...}

    Returns
    -------
        expo : dict
            community' user feature exposure
            {user1: {photo1: [transformed features], ...}, ...}

    """
    expo = {}
    for user, photos in users.items():
        expo[user], _, _ = user_expo(photos, f_top, detectors, opt_threshs, load_detectors, cfg)
    return expo
