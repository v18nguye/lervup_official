"""The module fine-tunes the focal rating parameters

"""
"""The module fine-tunes hyperameters

"""
import copy
import joblib
import numpy as np
import copy


def run_model(modelb, cfg):
    """run model training
    
    """
    # Load config and train model.
    modelb.load_cfg(cfg)
    modelb.train()

    # Model validation.
    val_score = modelb.test(modelb.x_val)

    return modelb, val_score


def ftune_model_fr(modelb, cfg):
    """Training models with different FR configuration.
    
    :param modelb: 
        initiated model
    
    :param cfg:
        configuration objects

    """
    best_model = None
    best_val_score = None

    cfg_list = []
    nb_jobs = 0

    Ks = [5, 10, 15, 20]
    GAMMAs = [0, 1, 2, 3]

    for gamma in GAMMAs:
        cfg.FE.GAMMA = gamma
        for k in Ks:
            cfg.FE.K = k
            nb_jobs += 1
            cfg_list.append(copy.deepcopy(cfg))

    results = joblib.Parallel(n_jobs=nb_jobs, pre_dispatch='all')(joblib.delayed(run_model)(copy.deepcopy(modelb), cfg_) for cfg_ in cfg_list)
    opt_index = np.argmax([x[1] for x in results])
    best_model = results[opt_index][0]
    best_val_score = results[opt_index][1]

    print(results)
    print('best val model score: ', best_val_score)
    return best_model, best_val_score