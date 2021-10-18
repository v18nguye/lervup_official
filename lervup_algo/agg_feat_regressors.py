"""
Aggregated-visual-concept-feature based regressors.

"""
import os
from os.path import dirname, abspath
root = dirname(dirname(abspath(__file__)))

import copy
import numpy as np
from sklearn.decomposition import PCA

from .tools.ftune_agg_reg import ftune_rf
from .vispel.config import get_cfg
from .lib.detectors.activator import activator
from .lib.corr.corr_type import pear_corr
from .lib.loader.data_loader import train_test, gt_user_expos, vis_concepts


def _loader(detector):
    """Data loader"""

    gt_expo_path = "dataset/lervup_data/gt_usr_exposure_v1.json"
    concept_path = "dataset/lervup_data/visual_concepts/"

    if detector == 'mobinet':
        data_path = "dataset/lervup_data/train_val_test_split_mobinet_v2.json"
        opt_thresh_path = "base_algo/base_opt/out/mobi_optimal_thres_situs_v2.txt"
    
    elif detector == 'rcnn':
        data_path = "dataset/lervup_data/train_val_test_split_rcnn_v2.json"
        opt_thresh_path = "base_algo/base_opt/out/rcnn_optimal_thres_situs_v2.txt"

    opt_thresh_path = os.path.join(root, opt_thresh_path)
    data_train, data_val, data_test = train_test(root, data_path)
    gt_expo = gt_user_expos(root, gt_expo_path)
    raw_concepts = vis_concepts(root, concept_path)
    ord_concept = {}  # viual concept ordering
    count = 0
    for _, concepts in raw_concepts.items():
        for concept, _ in concepts.items():
            ord_concept[concept] = count
            count += 1
        break
    
    return data_train['100'], data_val, data_test, gt_expo, raw_concepts, ord_concept, opt_thresh_path


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
    train, val, test, gt_expo, raw_concepts, ord_concept, opt_thresh_path = _loader(detector)
    for situ, concepts in raw_concepts.items():
        print('***********')
        print(situ)
        print('***********')

        val_results = []
        model_list = []
        X_test_list = []
        y_test_list = []

        for feat_size in [None, 32, 16, 8]:

            X_train, y_train = _features(train, gt_expo, situ, ord_concept, concepts, opt_thresh_path)
            X_val, y_val = _features(val, gt_expo, situ, ord_concept, concepts, opt_thresh_path)
            X_test, y_test = _features(test, gt_expo, situ, ord_concept, concepts, opt_thresh_path)

            if feat_size  == None:
                X_train_red =  X_train
                X_val_red = X_val
                X_test_red = X_test
            else:
                pca = PCA(n_components=feat_size, svd_solver='full')
                pca.fit(X_train)
                X_train_red = pca.transform(X_train)
                X_val_red = pca.transform(X_val)
                X_test_red = pca.transform(X_test)

            best_model, best_val = ftune_rf(cfg, X_train_red, X_val_red, y_train, y_val)
            model_list.append(copy.deepcopy(best_model))
            val_results.append(best_val)
            X_test_list.append(X_test_red)
            y_test_list.append(y_test)

        print(val_results)

        opt_index = np.argmax(val_results[1:])
        best_val_model = model_list[1:][opt_index]
        raw_feat_model = model_list[0]
        
        raw_feat_y_pred = raw_feat_model.predict(X_test_list[0])
        raw_feat_test_result = pear_corr(y_test_list[0], raw_feat_y_pred)

        y_pred = best_val_model.predict(X_test_list[1:][opt_index])
        test_result = pear_corr(y_test_list[1:][opt_index], y_pred)

        print('Val result -  raw: ',"{:.2f}".format(val_results[0]),' - pca: ',"{:.2f}".format(val_results[1:][opt_index]))
        print('Raw-feat regression: '+"{:.2f}".format(raw_feat_test_result))
        print('PCA-feat-reduction regression: '+"{:.2f}".format(test_result))


def setup(detector):
    """

    :param args:
    :return:
        cfg
    """
    cfg = get_cfg()
    if detector == 'rcnn':
        cfg_path = root + '/lervup_algo/configs/rcnn_rf_kmeans.yaml'
    elif detector == 'mobinet':
        cfg_path = root + '/lervup_algo/configs/mobi_rf_kmeans.yaml'
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