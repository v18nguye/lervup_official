"""
Aggregated-visual-concept-feature based regressors.

"""
import os
from os.path import dirname, abspath
root = dirname(dirname(abspath(__file__)))

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor as RFR

from .vispel.config import get_cfg
from .lib.detectors.activator import activator
from .lib.corr.corr_type import pear_corr
from .lib.loader.data_loader import train_test, gt_user_expos, vis_concepts


def _loader(detector):
    """Data loader"""

    gt_expo_path = "dataset/lervup_data/gt_usr_exposure_v1.json"
    concept_path = "dataset/lervup_data/visual_concepts/"

    if detector == 'mobinet':
        data_path = "dataset/lervup_data/train_test_split_mobinet_v1.json"
        opt_thresh_path = "base_algo/base_opt/out/mobi_optimal_thres_situs_v1.txt"
    
    elif detector == 'rcnn':
        data_path = "dataset/lervup_data/train_test_split_rcnn_v1.json"
        opt_thresh_path = "base_algo/base_opt/out/rcnn_optimal_thres_situs_v1.txt"

    opt_thresh_path = os.path.join(root, opt_thresh_path)
    data_train, data_test = train_test(root, data_path)
    gt_expo = gt_user_expos(root, gt_expo_path)
    raw_concepts = vis_concepts(root, concept_path)
    ord_concept = {}  # viual concept ordering
    count = 0
    for _, concepts in raw_concepts.items():
        for concept, _ in concepts.items():
            ord_concept[concept] = count
            count += 1
        break
    
    return data_train['100'], data_test, gt_expo, raw_concepts, ord_concept, opt_thresh_path


def _features(data, gt_expo, situ, ord_concept, concepts, opt_thresh_path):
    """Create user's features by aggregation"""
    detectors, opt_thres = activator(concepts, situ, \
                                     opt_thresh_path, True)
    expo = gt_expo[situ]

    X_feature = []
    y_target = []

    for user, photos in data.items():
        x_user = [0 for i in range(len(ord_concept.keys()))]
        for _, obs in photos.items():
            for concept_, scores in obs.items():
                if concept_ in detectors and concept_ in ord_concept:
                    for score_ in scores:
                        if score_ > float(opt_thres[concept_]):
                            x_user[ord_concept[concept_]] += score_ * detectors[concept_]

        X_feature.append(x_user)
        y_target.append(expo[user])

    X_feature = np.asarray(X_feature)
    y_target = np.asarray(y_target)

    return X_feature, y_target


def regress(cfg, detector):
    """Regressor"""
    train, test, gt_expo, raw_concepts, ord_concept, opt_thresh_path = _loader(detector)
    for situ, concepts in raw_concepts.items():
        print('***********')
        print(situ)
        print('***********')

        test_results = []

        for feat_size in [None, 32, 16, 8]:

            X_train, y_train = _features(train, gt_expo, situ, ord_concept, concepts, opt_thresh_path)
            X_test, y_test = _features(test, gt_expo, situ, ord_concept, concepts, opt_thresh_path)

            if feat_size  == None:
                X_train_red =  X_train
                X_test_red = X_test
            else:
                pca = PCA(n_components=feat_size, svd_solver='full')
                pca.fit(X_train)
                X_train_red = pca.transform(X_train)
                X_test_red = pca.transform(X_test)

            if cfg.SOLVER.CORR_TYPE == 'PEARSON':
                score_type = {cfg.SOLVER.CORR_TYPE: make_scorer(pear_corr, greater_is_better=True)}

            tuning_params = {'bootstrap': cfg.REGRESSOR.RF.BOOTSTRAP,
                             'max_depth': cfg.REGRESSOR.RF.MAX_DEPTH,
                             'max_features': cfg.REGRESSOR.RF.MAX_FEATURES,
                             'min_samples_leaf': cfg.REGRESSOR.RF.MIN_SAMPLES_LEAF,
                             'min_samples_split': cfg.REGRESSOR.RF.MIN_SAMPLES_SPLIT,
                             'n_estimators': cfg.REGRESSOR.RF.N_ESTIMATORS,
                             'random_state': [42]}

            model = GridSearchCV(RFR(), tuning_params, cv=cfg.FINE_TUNING.CV,
                                 scoring=score_type, refit=list(score_type.keys())[0],
                                 n_jobs=cfg.FINE_TUNING.N_JOBS)

            model.fit(X_train_red, y_train)
            y_pred = model.predict(X_test_red)

            test_results.append(pear_corr(y_pred, y_test))

        print('Raw-feat regression: '+"{:.2f}".format(test_results[0]))
        print('PCA-feat-reduction regression: '+"{:.2f}".format(max(test_results[1:])))


def setup(detector):
    """

    :param args:
    :return:
        cfg
    """
    cfg = get_cfg()
    if detector == 'rcnn':
        cfg_path = root + '/lervup_algo/configs/rf_kmeans_ft_rcnn_cv5.yaml'
    elif detector == 'mobinet':
        cfg_path = root + '/lervup_algo/configs/rf_kmeans_ft_mobi_cv5.yaml'
    else:
        raise ValueError('Detector '+str(detector)+' not deployed yet.')
    cfg.merge_from_file(cfg_path)

    return cfg


def agg_feat_regs(detector):
    """Regression with aggregated features
    
    :param detector: str
        mobinet or rcnn
    """
    # cfg setting
    print('#-----------------------------------#')
    print('# '+detector.upper()+ ' TEST CORR')
    print("#-----------------------------------#")
    cfg = setup(detector)
    regress(cfg, detector)