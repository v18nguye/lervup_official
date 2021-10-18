"""The module fine-tunes hyparameters's random-forest-based regressors 

"""
import numpy as np
import copy
import joblib

def rf_run_model(rf_modelb, rf_cfg):
    """run regression tranining
    
    """
    # Load config and train model.
    rf_modelb.load_cfg(rf_cfg)
    rf_modelb.train_vispel()

    # Model validation.
    rf_val_result = rf_modelb.test_vispel(rf_modelb.X_val)

    return rf_modelb, rf_val_result


def ftune_rf(rf_modelb, rf_cfg):
    """Random-Forest FineTunner

    :param modelb: 
        model base

    :param cfg: CFG
        configuration object.
    """
    rf_cfg_list = []
    rf_nb_jobs = 0

    for bootstrap in [ [True], [False]]:
        rf_cfg.REGRESSOR.RF.BOOTSTRAP = bootstrap

        for max_depth in [[5], [7]]:
            rf_cfg.REGRESSOR.RF.MAX_DEPTH = max_depth

            for max_features in [['auto']]:
                rf_cfg.REGRESSOR.RF.MAX_FEATURES = max_features

                for min_samples_leaf in [[2] , [4]]:
                    rf_cfg.REGRESSOR.RF.MIN_SAMPLES_LEAF = min_samples_leaf

                    for min_samples_split in [[2], [4]]:
                        rf_cfg.REGRESSOR.RF.MIN_SAMPLES_SPLIT = min_samples_split

                        for n_estimators in [[100], [150], [200]]:
                            rf_cfg.REGRESSOR.RF.N_ESTIMATORS = n_estimators

                        rf_nb_jobs += 1
                        rf_cfg_list.append(copy.deepcopy(rf_cfg))

    rf_results = joblib.Parallel(n_jobs=rf_nb_jobs, pre_dispatch='all')(joblib.delayed(rf_run_model)(copy.deepcopy(rf_modelb), rf_cfg_) for rf_cfg_ in rf_cfg_list)
    rf_opt_index = np.argmax([x[1] for x in rf_results])
    rf_best_model = rf_results[rf_opt_index][0]
    rf_best_val_score = rf_results[rf_opt_index][1]

    return rf_best_model, rf_best_val_score
    

def ftune_rf_cv(rf_modelb, rf_cfg):
    """Random-Forest FineTunner

    :param modelb: 
        model base

    :param cfg: CFG
        configuration object.
    """
    rf_cfg.REGRESSOR.RF.BOOTSTRAP = [ True, False]
    rf_cfg.REGRESSOR.RF.MAX_DEPTH = [7, 9]
    rf_cfg.REGRESSOR.RF.MAX_FEATURES = ['auto']
    rf_cfg.REGRESSOR.RF.MIN_SAMPLES_LEAF = [2]
    rf_cfg.REGRESSOR.RF.MIN_SAMPLES_SPLIT = [2]
    rf_cfg.REGRESSOR.RF.N_ESTIMATORS = [150, 220]
    rf_cfg.FINE_TUNING.N_JOBS = 8

    # Load config and train model.
    rf_modelb.load_cfg(rf_cfg)
    rf_modelb.train_vispel()

    # Model validation.
    rf_val_result = rf_modelb.test_vispel(rf_modelb.X_val)

    return rf_modelb, rf_val_result








