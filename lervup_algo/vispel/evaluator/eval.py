from exposure.exposure import community_expo
from detectors.activator import activator
from regressor.features import build_features
from corr.corr_type import pear_corr, kendall_corr


def vispel_evaluator(situ_name, clusteror, regressor, vis_concepts, eval_set, gt_eval_expos, cfg):
    """
    Evaluate the trained models

    :param clusteror:
    :param regressor:
    :param train_set:
    :param gt_situ_expos:
    :param cfg:
    :return:

    """
    # Construct active detectors
    detectors, opt_threds = activator(vis_concepts, situ_name,\
                                      cfg.DATASETS.PRE_VIS_CONCEPTS, cfg.DETECTOR.LOAD)
    # Photo exposures of users
    commu_expo_features = community_expo(eval_set, cfg.SOLVER.F_TOP,\
                                       detectors, opt_threds, cfg.DETECTOR.LOAD, cfg.SOLVER.FILTERING)

    reg_features, gt_expos = build_features(clusteror, commu_expo_features, gt_eval_expos, cfg)

    pred_expos = regressor.predict(reg_features)

    if cfg.SOLVER.CORR_TYPE == 'KENDALL':
        print('correlation: ', "{:.4f}".format(kendall_corr(pred_expos, gt_expos)))

    elif cfg.SOLVER.CORR_TYPE == 'PEARSON':
        print('correlation: ', "{:.4f}".format(pear_corr(pred_expos, gt_expos)))