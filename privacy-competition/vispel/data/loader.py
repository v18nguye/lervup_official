from detectors.activator import activator
from loader.data_loader import train_test, gt_user_expos, vis_concepts


def data_loader(root, cfg, situation, N=-1):
    """
    N: number of training profiles
    :return:
    """

    # Basic data-loader

    X_mini_batches, X_test_set = train_test(root, cfg.DATASETS.TRAIN_TEST_SPLIT)
    expos = gt_user_expos(root, cfg.DATASETS.GT_USER_EXPOS)
    concepts = vis_concepts(root, cfg.DATASETS.VIS_CONCEPTS)
    X_community = {}

    # Build community data

    for user, objects in X_mini_batches['100'].items():
        X_community[user] = objects

    for user, objects in X_test_set.items():
        X_community[user] = objects

    # select users for debug mode
    X_train_set = X_mini_batches['100']

    situ_gt_expos = expos[situation]
    situ_vis_concepts = concepts[situation]

    # Construct active detectors
    detectors, opt_threds = activator(concepts, situation,\
                                      cfg.DATASETS.PRE_VIS_CONCEPTS, cfg.DETECTOR.LOAD)

    return X_train_set, X_test_set, X_community, \
           situ_gt_expos, situ_vis_concepts, detectors, opt_threds
