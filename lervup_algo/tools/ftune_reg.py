"""The module fine-tunes hyparameters's random-forest-based regressors 

"""
import numpy as np
import copy

def ftune_rf_cv(rf_modelb, rf_cfg, args, X_train, X_val, gt_expos, detectors, opt_threds):
    """Random-Forest FineTunner

    :param modelb: 
        model base

    :param cfg: CFG
        configuration object.
    """
    rf_cfg.REGRESSOR.RF.BOOTSTRAP = [True]
    rf_cfg.REGRESSOR.RF.MAX_DEPTH = [5, 7]
    rf_cfg.REGRESSOR.RF.MAX_FEATURES = ['auto']
    rf_cfg.REGRESSOR.RF.MIN_SAMPLES_LEAF = [1, 3]
    rf_cfg.REGRESSOR.RF.MIN_SAMPLES_SPLIT = [3]
    rf_cfg.REGRESSOR.RF.N_ESTIMATORS = [100]
    rf_cfg.GRID_SEARCH.N_JOBS = 2

    # Load config and train model.
    rf_modelb.init_model(rf_cfg)
    rf_modelb.train_vispel(X_train, gt_expos, detectors, opt_threds)

    # Model validation.
    rf_val_result = rf_modelb.test_vispel(X_val, gt_expos, detectors, opt_threds)

    return rf_modelb, rf_val_result





