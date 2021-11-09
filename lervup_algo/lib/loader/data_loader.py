import os
from os.path import dirname, abspath
import json
import copy
from detectors.activator import activator

def train_test(root, path):
    """Load user profile train, test

    :param root: string
        current working absolute path

    :param path: string
        relative path to saved train vs test data

    :return:
           train_data: dict
                training mini-batches
                    {ratio: {user1: {photo1: {class1: [obj1, ...], ...}}, ...}, ...}, ...}

            test_data: dict
                test data
                    {user1: {photo1: {class1: [obj1, ...], ...}}, ...}, ...}

    """
    train_test_info = json.load(open(os.path.join(root, path)))
    test_data = train_test_info['test']
    train_data = train_test_info['train']
    val_data = train_test_info['val']

    return train_data, val_data, test_data


def gt_user_expos(root, path):
    """Load crowd-sourcing user exposure

    :param root:
    :param path:

    :return:
        gt_usr_expo : dict
            ground-truth user exposure by situation
                 {situ1: {user1: avg_score, ...}, ...}
    """
    gt_usr_expo = json.load(open(os.path.join(root, path)))

    return gt_usr_expo


def vis_concepts(root, path):
    """Load object situation under a dictionary form

    :param root: string
    :param path: string
        path to situations
    :param denormalization: boolean

    :return:
        class_situs : dict
            situation and its crowd-sourcing class exposure corr
                {situ1: {class1: score, ...}, ...}
    """
    class_situs = {}
    situs = os.listdir(os.path.join(root, path))


    for situ in situs:
        situ_key = situ.split('.')[0]
        class_situs[situ_key] = {}
        with open(os.path.join(root, path, situ)) as fp:

            lines = fp.readlines()
            for line in lines:
                parts = line.split(' ')
                class_ = parts[0]
                score = float(parts[1])
                class_situs[situ_key][class_] = score

    return class_situs


def data_loader(cfg, situation):
    """
    N: number of training profiles
    :return:
    """
    root = abspath(__file__).split('/lervup_algo/lib/')[0]
    # Basic data-loader
    X_train_set, X_val_set, X_test_set = train_test(root, cfg.DATASETS.TRAIN_TEST_SPLIT)
    expos = gt_user_expos(root, cfg.DATASETS.GT_USER_EXPOS)
    concepts = vis_concepts(root, cfg.DATASETS.VIS_CONCEPTS)

    situ_gt_expos = expos[situation]

    # Construct active detectors
    detectors, opt_threds = activator(concepts, situation, \
                                      os.path.join(root, cfg.DATASETS.PRE_VIS_CONCEPTS), cfg.DETECTOR.LOAD)
                                      
    return X_train_set, X_val_set, X_test_set, \
           situ_gt_expos, detectors, opt_threds
