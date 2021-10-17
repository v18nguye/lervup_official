"""
Aggregated vis-concept based regressor.

python agg_regressor.py --cfg rf_kmeans_ft_rcnn_cv5.yaml
python agg_regressor.py --cfg rcnn_rf_kmeans.yaml
"""
import os
import argparse
import numpy as np
import _init_paths
import random
from vispel.config import get_cfg
from sklearn.decomposition import PCA
from detectors.activator import activator
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from corr.corr_type import pear_corr
from sklearn.ensemble import RandomForestRegressor as RFR
from loader.data_loader import train_test, gt_user_expos, vis_concepts

root = os.getcwd().split('/privacy/tools')[0]
data_path = "process_raw_data/out/train_test_split_mobinet_v1.json"
gt_expo_path = "process_raw_data/out/gt_usr_exposure_v1.json"
concept_path = "process_raw_data/raw_data/visual_concepts/processed_situations"
opt_thresh_path = "privacy_baseline/out/mobi_optimal_thres_situs_v1.txt"


def set_seeds(seed_=2020):
    random.seed(seed_)
    np.random.seed(seed_)

def _loader():
    """Data loader"""

    data_train, data_test = train_test(root, data_path)
    gt_expo = gt_user_expos(root, gt_expo_path)
    raw_concepts = vis_concepts(root, concept_path)
    ord_concept = {}  # order of concept
    count = 0
    for situ, concepts in raw_concepts.items():
        for concept, _ in concepts.items():
            ord_concept[concept] = count
            count += 1
        break

    return data_train['100'], data_test, gt_expo, raw_concepts, ord_concept


def _features(data, gt_expo, situ, ord_concept, concepts):
    """Create user feature"""
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


def regress(cfg):
    """Regressor"""
    train, test, gt_expo, raw_concepts, ord_concept = _loader()
    for situ, concepts in raw_concepts.items():
        print('***********')
        print(situ)
        print('***********')

        for compenent_ in [None, 16, 8, 4]:

            X_train, y_train = _features(train, gt_expo, situ, ord_concept, concepts)
            X_test, y_test = _features(test, gt_expo, situ, ord_concept, concepts)

            if compenent_  == None:
                X_train_red =  X_train
                X_test_red = X_test
            else:
                pca = PCA(n_components=compenent_, svd_solver='full', iterated_power=10, random_state=42)
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
            if compenent_ == None:
                compenent_ = 282
            print('N= ' + str(compenent_) + ': ' + str(pear_corr(y_pred, y_test)))


def argument_parser():
    """
    Create a parser with some common arguments.

    Returns:
        argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--model_name", required=True, help="saved modeling name")
    parser.add_argument("--fe", required=True, help="1 to turn on the focal exposure")
    parser.add_argument("--situation", required=True, help="IT, ACCOM, BANK, WAIT")
    parser.add_argument("--N", required=True, help="Number of training profiles: "
                                                   "-1: ALL user profiles"
                                                   "N: N profiles (N < 400)")
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )

    return parser


def argument_parser():
    """
    Create a parser with some common arguments.

    Returns:
        argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="", metavar="FILE", help="path to config file")

    return parser


def set_up(args):
    """

    :param args:
    :return:
        cfg
    """
    # set_seeds()
    cfg = get_cfg()
    config = os.path.join(root, 'privacy/configs/', args.cfg)
    print(config)
    cfg.merge_from_file(config)

    return cfg


if __name__ == '__main__':
    args = argument_parser().parse_args()
    cfg = set_up(args)

    regress(cfg)
