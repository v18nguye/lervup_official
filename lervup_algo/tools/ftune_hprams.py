"""The module fine-tunes hyperameters

"""
import copy
import joblib
import numpy as np
from .ftune_reg import ftune_rf_cv
from loader.data_loader import data_loader

def run_model(modelb, cfg, args):
    """run model training
    
    """
    X_train, X_val, _, \
        gt_expos, \
        detectors, opt_threds = data_loader(cfg, modelb.situ_name)

    trained_model, val_score = ftune_rf_cv(modelb, cfg, args, X_train, X_val, gt_expos, detectors, opt_threds)
    
    return trained_model, val_score


def ftune_model_hyp(modelb, cfg, args):
    """Training models with different hyparam configuration.
    
    :param modelb: 
        initiated model
    
    :param cfg:
        configuration objects

    """
    cfg_list = []

    EPSs = [0.05, 0.1, 0.15]
    KEEPs = [0.87,0.94, 1.0]
    REGRESSOR_FEATURES = ['FR1', 'FR2']
    DETECTOR_LOADs = [True, False]
    Ks = [10, 15, 20]
    GAMMAs = [0, 2, 4]
    TAU_es = [0.8, 1, 1.333]
    FE_MODEs = ['IMAGE', 'OBJECT']
    CLUSTERs = [1, 2, 8]

    for cluster in CLUSTERs:
        cfg.CLUSTEROR.K_MEANS.CLUSTERS = cluster
        for eps in EPSs:
            cfg.USER_SELECTOR.EPS = eps
            for keep in KEEPs:
                cfg.USER_SELECTOR.KEEP = keep
                for load in DETECTOR_LOADs:
                    cfg.DETECTOR.LOAD = load
                    for rfeat in REGRESSOR_FEATURES:
                        cfg.REGRESSOR.FEATURES = rfeat
                        for fe_mode in FE_MODEs:
                            cfg.FE.MODE = fe_mode
                            for gamma in GAMMAs:
                                cfg.FE.GAMMA = gamma
                                for k in Ks:
                                    cfg.FE.K = k
                                    for tau_e in TAU_es:
                                        cfg.FE.TAU_e = tau_e
                                        cfg_list.append(copy.deepcopy(cfg))

    results = joblib.Parallel(n_jobs=10)(joblib.delayed(run_model)(copy.deepcopy(modelb), cfg_, args) for cfg_ in cfg_list)
    opt_index = np.argmax([x[1] for x in results])
    best_model = results[opt_index][0]
    best_val_score = results[opt_index][1]

    # print(results)
    print('best model score: ', best_val_score)
    return best_model, best_val_score