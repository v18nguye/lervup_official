import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler, MaxAbsScaler

def agg_features(com_features):
    """Aggregate exposure features of all images
        in the community.

    :param: com_features : dict
        photos and it features
           {user1: {photo1: [transformed features], ...}, ...}

    Returns
    -------
        transformed_features : list
            [ [user1's photo1's transformed features], ...]

    """
    features = []

    for user, user_features in com_features.items():
        for photo, photo_features in user_features.items():
            features.append(photo_features)
    features = np.asarray(features)

    return features


def train_clusteror(situ_name, model, com_features, cfg):
    """
    Train clusteror on all images of the community, which will be used
    further to cluster each user's image.

    Parameters
    ----------
    model: object
        clusteror modeling

    com_features : dict
        community exposure features
        dict of all users in a given situation with their clusteror features
            {user1: {photo1:[transformed features], ...}, ...}

    """
    aggfeatures_ = agg_features(com_features)

    clus_feat_transform = RobustScaler().fit(aggfeatures_)
    aggfeatures_transform = clus_feat_transform.transform(aggfeatures_)

    if cfg.CLUSTEROR.TYPE == 'K_MEANS':
        model.fit(aggfeatures_transform)
    elif cfg.CLUSTEROR.TYPE == 'GM':
        model.fit(aggfeatures_transform)

    return model, clus_feat_transform