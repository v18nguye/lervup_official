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
        A top N ranked detection object confidence

    load_detectors : boolean
        load active detectors pre-computed by the privacy
        base-line method

    detectors : dict
        active detectors in a given situation and its score
            {detector1: score, ...}


    opt_threds:
            optimal threshold for each object. Precomputed by the base line privacy method.
    Returns
    -------
        expo_obj : tuple
            photo exposure and its objectness sum
                {exp +, expo -, objness}
    """

    expo_pos = []  # positive exposure
    expo_neg = []  # negative exposure
    objectness = []

    attract_pos_concepts = []
    attract_neg_concepts = []
    neutral_pos_concepts = []
    neutral_neg_concepts = []
    scale_flag = False

    for object_, scores in photo.items():
        obj_score = 0

        if object_ in detectors:
            # Statistic on extreme concepts
            if detectors[object_] > cfg.FE.TAU_e:
                attract_pos_concepts.append(object_)
            elif detectors[object_] < -cfg.FE.TAU_e:
                attract_neg_concepts.append(object_)

            if 0 <= detectors[object_] <= cfg.FE.TAU_e:
                neutral_pos_concepts.append(object_)
            if -cfg.FE.TAU_e <= detectors[object_] < 0:
                neutral_neg_concepts.append(object_)

            # Estimate object-ness of the image
            if not load_detectors:
                valid_obj = [score for score in scores if score >= f_top]
                if sum(valid_obj) > 0:
                    obj_score += sum(valid_obj) / len(valid_obj)
            else:
                valid_obj = [score for score in scores if score >= opt_threshs[object_]]
                if sum(valid_obj) > 0:
                    obj_score += sum(valid_obj) / len(valid_obj)
            objectness.append(obj_score)

    if sum(objectness) != 0:
        objectness = sum(objectness) / len(objectness)
    else:
        objectness = 0

    # Apply Focal Rating

    if cfg.FE.MODE == 'IMAGE':

        if len(neutral_pos_concepts) != 0:

            ratio = len(attract_pos_concepts) / (len(neutral_pos_concepts) + len(attract_pos_concepts))
            if 0 < ratio < cfg.FE.TAU_o:
                scale_pos_flag = True
            else:
                scale_pos_flag = False
        else:
            scale_pos_flag = False

        if len(neutral_neg_concepts) != 0:
            ratio = len(attract_neg_concepts) / (len(neutral_neg_concepts) + len(attract_neg_concepts))
            if 0 < ratio < cfg.FE.TAU_o:
                scale_neg_flag = True
            else:
                scale_neg_flag = False
        else:
            scale_neg_flag = False

    elif cfg.FE.MODE == 'OBJECT':
        scale_pos_flag = True
        scale_neg_flag = True

    for object_, scores in photo.items():

        obj_score = 0
        if object_ in detectors:
            if not load_detectors:
                valid_obj = [score for score in scores if score >= f_top]
                if sum(valid_obj) > 0:
                    obj_score += sum(valid_obj) / len(valid_obj)
            else:
                valid_obj = [score for score in scores if score >= opt_threshs[object_]]
                if sum(valid_obj) > 0:
                    obj_score += sum(valid_obj) / len(valid_obj)

            # Only scale object scores when object-ness is sufficiently high, and
            # exist numerous neutral objects as the same type (positive or negative)
            if detectors[object_] >= 0:
                if scale_pos_flag and objectness > cfg.FE.TAU_i:
                    scale_flag = True
                    scaled_expo = FE(detectors[object_], cfg.FE.GAMMA, cfg.FE.K)
                else:
                    scaled_expo = detectors[object_]
                expo_pos.append(scaled_expo)

            if detectors[object_] < 0:
                if scale_neg_flag and objectness > cfg.FE.TAU_i:
                    scale_flag = True
                    scaled_expo = FE(detectors[object_], cfg.FE.GAMMA, cfg.FE.K)
                else:
                    scaled_expo = detectors[object_]
                expo_neg.append(scaled_expo)

    if sum(expo_pos) != 0:
        expo_pos = sum(expo_pos) / len(expo_pos)
    else:
        expo_pos = 0

    if sum(expo_neg) != 0:
        expo_neg = sum(expo_neg) / len(expo_neg)
    else:
        expo_neg = 0
    expo_obj = (expo_pos, expo_neg, objectness)

    return expo_obj, scale_flag


def pooled_expo(photo, f_top, detectors, opt_threshs, load_detectors, cfg):
    """
    Apply max-pooling method to calculate photo exposures

    Parameters
    ----------
    photo
    f_top
    detectors
    opt_threshs
    load_detectors
    cfg

    Returns
    -------

    """
    obj_ness = []
    concepts = []

    for object_, scores in photo.items():

        if object_ in detectors:

            # Estimate object-ness of the image
            if not load_detectors:
                valid_obj = [score for score in scores if score >= f_top]
            else:
                valid_obj = [score for score in scores if score >= opt_threshs[object_]]

            if sum(valid_obj) > 0:

                if cfg.SOLVER.PFT == 'POOLINGx2':
                    obj_ness.append(max(valid_obj))

                elif cfg.SOLVER.PFT == 'POOLING':
                    obj_ness.append(sum(valid_obj) / len(valid_obj))
            else:
                obj_ness.append(0)

            concepts.append(detectors[object_])

    if len(concepts) > 0:

        pos_concepts = []
        neg_concepts = []
        obj_img = []
        max_index = np.argmax(np.abs(concepts))
        max_concept = concepts[max_index]

        for k in range(len(concepts)):
            if abs(concepts[k]) == abs(max_concept):
                scaled_concept = FE(concepts[k], cfg.FE.GAMMA, cfg.FE.K)
                obj_img.append(obj_ness[k])

                if concepts[k] > 0:
                    pos_concepts.append(scaled_concept)
                else:
                    neg_concepts.append(scaled_concept)

        if len(pos_concepts) > 0:
            pos_ = sum(pos_concepts) / len(pos_concepts)
        else:
            pos_ = 0

        if len(neg_concepts) > 0:
            neg_ = sum(neg_concepts) / len(neg_concepts)
        else:
            neg_ = 0

        if len(obj_img) > 0:
            obj_ = sum(obj_img) / len(obj_img)
        else:
            obj_ = 0

        expo_obj = (pos_, neg_, obj_)
        scale_flag = True

    else:
        expo_obj = (0, 0, 0)
        scale_flag = False

    return expo_obj, scale_flag


def user_expo(user_photos, f_top, detectors, opt_threshs, load_detectors, cfg):
    """Estimate user exposure

    Parameters
    ----------
        user_photos : dict
            user photos associating with predicted object confidence
                {photo1: {class1: [obj1, ...], ...},...}

        load_detectors : boolean
            load active detectors pre-computed by the privacy
            base-line method

        f_top : float [0,1)
            A top N ranked detected object confidence

        detectors : dict
            active detectors for a given situation

        filter : boolean
            filtering neutral photos with a threshold 0.01
        
    Returns
    -------
        expo : dict
            user exposure
                {photo1: [transformed features],...}

    """
    expo = {}
    count_rescaled_imgs = []
    for photo in user_photos:
        if cfg.SOLVER.PFT == 'ORG':
            (pos_expo, neg_expo, objectness), scale_flag = photo_expo(user_photos[photo], f_top, detectors, opt_threshs,
                                                                      load_detectors, cfg)

        elif cfg.SOLVER.PFT == 'POOLING' or cfg.SOLVER.PFT == 'POOLINGx2':
            (pos_expo, neg_expo, objectness), scale_flag = pooled_expo(user_photos[photo], f_top, detectors,
                                                                       opt_threshs,
                                                                       load_detectors, cfg)

        f_expo_pos = pos_expo
        f_expo_neg = neg_expo
        f_dens = objectness

        if scale_flag:
            count_rescaled_imgs.append(photo)

        if cfg.SOLVER.FILTERING:
            if abs(f_expo_pos) + abs(f_expo_neg) >= cfg.SOLVER.FILT:
                # Apply feature transform
                expo[photo] = feature_transform(f_expo_pos, f_expo_neg, f_dens, cfg.SOLVER.FEATURE_TYPE)
        else:
            # Apply feature transform
            expo[photo] = feature_transform(f_expo_pos, f_expo_neg, f_dens, cfg.SOLVER.FEATURE_TYPE)

    return expo, len(count_rescaled_imgs), len(user_photos)


def community_expo(users, f_top, detectors, opt_threshs, load_detectors, cfg):
    """Estimate photo exposure for all users in a given situation

    Parameters
    ----------
        users : dict
            users and their photos
                {user1: {photo1: {class1: [obj1, ...], ...}, ...}, ...}
        
        f_top : float [0,1)
            A top N ranked object detection confidence

        load_detectors : boolean
            load active detectors pre-computed by the privacy
            base-line method

        detectors : dict
            active detectors for a given situation

        opt_threshs : dict
            optimal active detector thresholds
                {detector1: thresh1, ...}

    Returns
    -------
        expo : dict
            community exposure
            {user1: {photo1: [transform features], ...}, ...}

    """
    expo = {}
    scaled = 0
    total = 0
    for user, photos in users.items():
        expo[user], scaled_images, user_images = user_expo(photos, f_top, detectors, opt_threshs, load_detectors, cfg)
        scaled += scaled_images
        total += user_images

    return expo
