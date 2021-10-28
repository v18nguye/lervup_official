"""The module fine-tunes hyperameters

"""
import copy
import joblib
import numpy as np
from .ftune_reg import ftune_rf, ftune_rf_cv


def run_model(modelb, cfg):
    """run model training
    
    """
    trained_model, val_score = ftune_rf_cv(modelb, cfg)

    return trained_model, val_score


def ftune_model_hyp(modelb, cfg, args):
    """Training models with different hyparam configuration.
    
    :param modelb: 
        initiated model
    
    :param cfg:
        configuration objects

    """
    cfg_list = []
    nb_jobs = 0

    SOLVER_FEATURE_TYPEs = ['ORG']
    DETECTOR_LOADs = [True]
    F_TOPs = [0.1]
    FILTs = [0.05]

    if args.fr > 0: # use focal rating
        Ks = [10, 20]
        GAMMAs = [0, 1, 3]
        TAU_es= [0.7, 10]
        FE_MODEs = ['IMAGE', 'OBJECT']
        PFT_MODEs = ['ORG', 'POOLING', 'POOLINGx2']

    else:
        Ks = [10]
        GAMMAs = [0] # no scaling
        TAU_es= [0] # no scaling
        FE_MODEs = ['IMAGE']
        PFT_MODEs = ['ORG']

    for ftype in SOLVER_FEATURE_TYPEs:
        cfg.SOLVER.FEATURE_TYPE = ftype
        for fe_mode in FE_MODEs:
            cfg.FE.MODE = fe_mode
            for tau_e in TAU_es:
                cfg.FE.TAU_e = tau_e
                for mode in PFT_MODEs:
                    cfg.SOLVER.PFT = mode
                    for load in DETECTOR_LOADs:
                        cfg.DETECTOR.LOAD = load
                        for f_top in F_TOPs:
                            cfg.SOLVER.F_TOP = f_top
                            for gamma in GAMMAs:
                                cfg.FE.GAMMA = gamma
                                for filt in FILTs:
                                    cfg.SOLVER.FILT = filt
                                    for k in Ks:
                                        cfg.FE.K = k
                                        nb_jobs += 1
                                        cfg_list.append(copy.deepcopy(cfg))

    results = joblib.Parallel(n_jobs=25)(joblib.delayed(run_model)(copy.deepcopy(modelb), cfg_) for cfg_ in cfg_list)
    print(results)
    opt_index = np.argmax([x[1] for x in results])
    best_model = results[opt_index][0]
    best_val_score = results[opt_index][1]

    print('best model score: ', best_val_score)
    return best_model, best_val_score