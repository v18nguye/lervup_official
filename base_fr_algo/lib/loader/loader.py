import os
from .data_loader import load_train_test, load_gt_user_expo, load_situs


def bloader(root, cfg, situ):

    """
    Data loader for the baseline

    Parameters
    ----------
    cfg: object
        object config

    situ: string
        considered situation

    Returns
    -------

    """
    expo_path = os.path.join(root, cfg.DATASETS.GT_USER_EXPOS)
    data_path = os.path.join(root, cfg.DATASETS.TRAIN_TEST_SPLIT)
    concept_path = os.path.join(root, cfg.DATASETS.VIS_CONCEPTS)

    gt_expos = load_gt_user_expo(expo_path)[situ]
    detectors = load_situs(concept_path)[situ]
    mini_batches, test_data = load_train_test(data_path)

    x_train = mini_batches['100']
    x_test = test_data

    return x_train, x_test, detectors, gt_expos
