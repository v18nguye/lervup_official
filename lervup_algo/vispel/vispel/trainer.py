from sklearn.decomposition import PCA
from exposure.exposure import community_expo
from clusteror.clustering import train_clusteror
from regressor.features import build_features, user_selector
from regressor.regression import train_regressor


def trainer(situ_name, X_train_set, gt_situ_expos, clusteror, regressor, detectors, opt_threds,  cfg):
    """
    Train an visual privacy exposure predictor on a situation

    Parameters
    ----------
    situ_name
    X_train_set:
        user ids in the train set
    gt_situ_expos
    clusteror
    regressor
    detectors:
        activated detectors
    opt_threds:
        optimal thresholds for visual detectors
    cfg
    Returns
    -------

    """
    # Calculate photo exposures
    # for user ids in the train set
    train_expo_features = community_expo(X_train_set, cfg.SOLVER.F_TOP,\
                                       detectors, opt_threds, cfg.DETECTOR.LOAD, cfg)

    # Build exposure features for  all user ids
    # by clustering their photo exposures
    trained_clusteror, clus_feat_transform = train_clusteror(situ_name, clusteror, train_expo_features, cfg)

    # Build regression features for user ids in the trained set
    reg_train_features, gt_train_expos = build_features(trained_clusteror, clus_feat_transform, train_expo_features, gt_situ_expos, cfg)

    # Select pertinent users
    reg_train_features, gt_train_expos = user_selector(reg_train_features, gt_train_expos, cfg)

    # Feature selector (feature reduction)
    if cfg.PCA.APPLY:
        pca = PCA(n_components=cfg.PCA.N_COMPONENTS)
        pca.fit(reg_train_features)
        X_train_rd = pca.transform(reg_train_features)

    else:
        pca = None
        X_train_rd = reg_train_features

    # Fit to the regressor
    trained_regressor, reg_feat_transform = train_regressor(regressor, X_train_rd, gt_train_expos, cfg)

    return clus_feat_transform, reg_feat_transform, pca