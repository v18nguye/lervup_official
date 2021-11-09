import numpy as np
from numpy import linalg as LA

def user_selector(reg_features, gt_expos, cfg):
    """
    Select pertinent users in the train set

    Parameters
    ----------
    reg_features
    gt_expos
    cfg

    Returns
    -------

    """
    if cfg.USER_SELECTOR.STATE:

        sty_dists = [] # similarity distance

        for k in range(reg_features.shape[0]):
            ref_expo = gt_expos[k]
            user_dists = []
            for h in range(reg_features.shape[0]):
                if h != k:
                    cur_expo =gt_expos[h]
                    if abs(cur_expo - ref_expo) < cfg.USER_SELECTOR.EPS:
                        feature_dist = LA.norm(reg_features[k,:]-reg_features[h,:])
                        user_dists.append(feature_dist)

            if len(user_dists) > 0:
                sty_dists.append(sum(user_dists)/len(user_dists))
            else:
                sty_dists.append(0)

        n_users = int(cfg.USER_SELECTOR.KEEP*reg_features.shape[0])
        sty_dists = np.asarray(sty_dists)
        sorted_indexes = np.argsort(sty_dists)
        sorted_gt_expos = gt_expos[sorted_indexes][:n_users]
        sorted_reg_features = reg_features[sorted_indexes, :][:n_users,:]

    else:
        sorted_reg_features = reg_features
        sorted_gt_expos = gt_expos

    return sorted_reg_features, sorted_gt_expos


def user_features(clusteror, clus_transformer, user_expo_features, cfg):
    """
    Build user regression features

    :param clus_transformer:
        feature transformer for clustering (normalize, etc, ..)
    :param user_expo_features:
    :param cfg:

    :return:
    """
    reg_features = []
    agg_features = []

    for photo, expo_features in user_expo_features.items():
        agg_features.append(expo_features)

    agg_features = np.asarray(agg_features)
    agg_features = clus_transformer.transform(agg_features)

    if cfg.CLUSTEROR.TYPE == 'K_MEANS':
        N_COMPONENTS = cfg.CLUSTEROR.K_MEANS.CLUSTERS
        photo_labels = clusteror.predict(agg_features)
        centroids = clusteror.cluster_centers_

    elif cfg.CLUSTEROR.TYPE == 'GM':
        N_COMPONENTS = cfg.CLUSTEROR.GM.COMPONENTS
        photo_labels = clusteror.predict(agg_features)
        centroids = clusteror.means_

    for k in range(N_COMPONENTS):
        photo_indexes = np.where(photo_labels == k)[0]

        if cfg.REGRESSOR.FEATURES == 'FR1':
            if len(photo_indexes) > 0:
                cluster_expo_features = agg_features[photo_indexes, :]
                centroid = centroids[k, :]

                mean_ = np.mean(cluster_expo_features, 0)
                distance = LA.norm(mean_ - centroid, ord=2)
                cluster_varnorms = np.var(cluster_expo_features, 0) # use variance instead of NORM as the old methode.

                # cluster_varnorms = LA.norm(cluster_expo_features, 'fro')
            else:
                cluster_varnorms =  np.zeros(centroids.shape[1])
                distance = 0

                # centroid = np.zeros(centroids.shape[1]) # there are no photos belong
                #                         # to the current centroid k
                # cluster_varnorms = 0

            for x in list(cluster_varnorms):
                reg_features.append(x)
            reg_features.append(distance)

            # for x in list(centroid):
            #     reg_features.append(x)
            # reg_features.append(cluster_varnorms)


        elif cfg.REGRESSOR.FEATURES == 'FR2':

            if len(photo_indexes) > 0:
                cluster_expo_features = agg_features[photo_indexes, :]
                mean_ = np.mean(cluster_expo_features, 0)
                cluster_varnorms = np.var(cluster_expo_features, 0) # use variance instead of NORM as the old methode.
                # cluster_varnorms = LA.norm(cluster_expo_features, 'fro')
            else:
                mean_ = np.zeros(centroids.shape[1]) # there are no photos belong
                                                        # to the current centroid k
                cluster_varnorms = np.zeros(centroids.shape[1])
                # cluster_norms = 0
                
            for x, y in zip(list(mean_), list(cluster_varnorms)):
                reg_features.append(x)
                reg_features.append(y)

            # for x, in list(mean_):
            #     reg_features.append(x)
            # reg_features.append(cluster_varnorms)
    return reg_features


def build_features(clusteror, clus_transformer, com_features, gt_situ_expos, cfg):
    """Build regression features for all users
        in the community.

    :param: com_features : dict
        community exposure features
        dict of all users in a given situation with their clusteror features
            {user1: {photo1:[transformed features], ...}, ...}

    :param: clus_transformer:
        feature transformer for clustering

    :param: clusteror: object
        trained clusteror on a given situation

    :param: cfg

    :param: gt_situ_expos: dict
        ground-truth user exposures in a given situation

    Returns
    -------
        X_features: numpy format [Number user x Number user's features]
        y_targets: numpy format [Number user x 1]

        regression_feature_users : dict
            {user1: [feature1,...], ...}

    """
    regression_features = []
    regression_targets = []
    non_im_users = []   # users without valid images
    for user, im_dict in com_features.items():
        if len(im_dict) == 0:
            non_im_users.append(user)

    for user, user_expo_features in com_features.items():
        if user not in non_im_users:
            regression_features.append(user_features(clusteror, clus_transformer,\
                                                     user_expo_features, cfg))
            regression_targets.append(gt_situ_expos[user])

    X_features = np.asarray(regression_features)
    y_targets = np.asarray(regression_targets)
    return X_features, y_targets