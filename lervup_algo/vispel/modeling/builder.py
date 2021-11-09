from sklearn.svm import SVR
from sklearn.cluster import KMeans
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.mixture import BayesianGaussianMixture as GM

from corr.corr_type import pear_corr, kendall_corr

def clusteror_builder(cfg):
    """
    Build a clusteror

    :return:
    """
    if cfg.CLUSTEROR.TYPE == 'K_MEANS':
        model = KMeans(n_clusters= cfg.CLUSTEROR.K_MEANS.CLUSTERS,\
                       n_init = cfg.CLUSTEROR.K_MEANS.N_INIT,
                       max_iter = cfg.CLUSTEROR.K_MEANS.MAX_ITER,
                       algorithm = cfg.CLUSTEROR.K_MEANS.ALGORITHM,
                       random_state=cfg.MODEL.SEED)

    elif cfg.CLUSTEROR.TYPE == 'GM':
        model = GM(n_components=cfg.CLUSTEROR.GM.COMPONENTS, \
                    covariance_type=cfg.CLUSTEROR.GM.COV_TYPE,
                    max_iter=cfg.CLUSTEROR.GM.MAX_ITER,
                    random_state=cfg.MODEL.SEED)

    return model


def regressor_builder(cfg):
    """
    Build SVM or RF modeling with pre-defined parameters.

    :param cfg:
    :return:

    """

    if not cfg.GRID_SEARCH.STATUS:

        if cfg.REGRESSOR.TYPE == 'SVM':
            model = SVR(kernel= cfg.REGRESSOR.SVM.KERNEL[0], C= cfg.REGRESSOR.SVM.C[0], 
                        gamma= cfg.REGRESSOR.SVM.GAMMA[0], random_state = cfg.MODEL.SEED)

        if cfg.REGRESSOR.TYPE == 'RF':
            model = RFR(bootstrap = cfg.REGRESSOR.RF.BOOTSTRAP[0], max_depth=cfg.REGRESSOR.RF.MAX_DEPTH[0],\
                        max_features = cfg.REGRESSOR.RF.MAX_FEATURES[0], min_samples_leaf=cfg.REGRESSOR.RF.MIN_SAMPLES_LEAF[0],\
                        min_samples_split = cfg.REGRESSOR.RF.MIN_SAMPLES_SPLIT[0], n_estimators=cfg.REGRESSOR.RF.N_ESTIMATORS[0],\
                        random_state = cfg.MODEL.SEED)

    else:

        if cfg.SOLVER.CORR_TYPE == 'KENDALL':
            score_type = {cfg.SOLVER.CORR_TYPE: make_scorer(kendall_corr, greater_is_better=True)}

        if cfg.SOLVER.CORR_TYPE == 'PEARSON':
            score_type = {cfg.SOLVER.CORR_TYPE: make_scorer(pear_corr, greater_is_better=True)}

        if cfg.REGRESSOR.TYPE == 'SVM':
            tuning_params = {'kernel': cfg.REGRESSOR.SVM.KERNEL,
                                  'gamma': cfg.REGRESSOR.SVM.GAMMA,
                                  'C': cfg.REGRESSOR.SVM.C,
                                  'random_state': [cfg.MODEL.SEED]}
                                  
            model = GridSearchCV(SVR(), tuning_params, cv = cfg.GRID_SEARCH.CV,
                                    scoring= score_type, refit=list(score_type.keys())[0],
                                    n_jobs= cfg.GRID_SEARCH.N_JOBS)

        if cfg.REGRESSOR.TYPE == 'RF':
            tuning_params = {'bootstrap': cfg.REGRESSOR.RF.BOOTSTRAP,
                                'max_depth': cfg.REGRESSOR.RF.MAX_DEPTH,
                                'max_features': cfg.REGRESSOR.RF.MAX_FEATURES,
                                'min_samples_leaf': cfg.REGRESSOR.RF.MIN_SAMPLES_LEAF,
                                'min_samples_split': cfg.REGRESSOR.RF.MIN_SAMPLES_SPLIT,
                                'n_estimators': cfg.REGRESSOR.RF.N_ESTIMATORS,
                                'random_state': [cfg.MODEL.SEED]}

            model = GridSearchCV(RFR(), tuning_params, cv= cfg.GRID_SEARCH.CV,
                                    scoring=score_type, refit=list(score_type.keys())[0],
                                    n_jobs= cfg.GRID_SEARCH.N_JOBS)

    return model