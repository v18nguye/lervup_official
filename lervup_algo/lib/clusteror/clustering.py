import numpy as np
import matplotlib.pyplot as plt

def agg_features(com_features, train = True):
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

            if train:
                features.append([abs(feature) for feature in photo_features])
            else:
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
    aggfeatures_ = agg_features(com_features, train= False)

    if cfg.CLUSTEROR.TYPE == 'K_MEANS':
        model.fit(aggfeatures_)
        centers = model.cluster_centers_
        labels = model.labels_

    elif cfg.CLUSTEROR.TYPE == 'GM':
        model.fit(aggfeatures_)
        centers = model.means_
        labels = model.predict(aggfeatures_)

    if cfg.MODEL.PLOT:

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(aggfeatures_[:,0], aggfeatures_[:,1], c=labels, s=2)
        for i, j in centers:
            ax.scatter(i, j, s=50, c='red', marker='+')
        ax.set_xlabel('object-ness')
        ax.set_ylabel('expo_score')

        fig.savefig(situ_name+'.jpg')

    return model

def test_clusteror(situ_name, trained_clusteror, test_features, cfg):
    """
    Test clusteror on all images of the test set.

    Parameters
    ----------
    model: object
        trained_clusteror modeling

    test_features : dict
        community exposure features
        dict of all users in a given situation with their clusteror features
            {user1: {photo1:[transformed features], ...}, ...}

    """
    aggfeatures_ = agg_features(test_features, train = False)

    if cfg.CLUSTEROR.TYPE == 'K_MEANS':
        centers = trained_clusteror.cluster_centers_
        labels = trained_clusteror.predict(aggfeatures_)

    elif cfg.CLUSTEROR.TYPE == 'GM':
        centers = trained_clusteror.means_
        labels = trained_clusteror.predict(aggfeatures_)

    if cfg.MODEL.PLOT:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(aggfeatures_[:,0], aggfeatures_[:,1], aggfeatures_[:,2], c=labels, s=3)

        for i, j, z in centers:
            ax.scatter(i, j, z, s=50, c='red', marker='+')


        ax.set_xlabel('fa')
        ax.set_ylabel('fn')
        ax.set_zlabel('fo')

        fig.savefig(situ_name+'_test.svg', format='svg', dpi=1200)