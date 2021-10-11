import os
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
    if cfg.MODEL.DEBUG:
        X_train_set = X_mini_batches['30']  # 30 % training users
    else:
        if int(N) == -1:
            X_train_set = X_mini_batches['100']
        else:
            X_all = X_mini_batches['100']
            count = 0
            X_train_set = {}
            for user_, photos_ in X_all.items():
                count += 1
                if count <= int(N):
                    X_train_set[user_] = photos_
                else:
                    break

    situ_gt_expos = expos[situation]
    situ_vis_concepts = concepts[situation]

    # Construct active detectors
    detectors, opt_threds = activator(concepts, situation, \
                                      os.path.join(root, cfg.DATASETS.PRE_VIS_CONCEPTS), cfg.DETECTOR.LOAD)
    return X_train_set, X_test_set, X_community, \
           situ_gt_expos, situ_vis_concepts, detectors, opt_threds
