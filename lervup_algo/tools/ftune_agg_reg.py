"""The module fine-tunes hyparameters's random-forest-based regressors 
for aggregated-feature-based method.

"""
import numpy as np
import copy
import joblib
from sklearn.ensemble import RandomForestRegressor as RFR
from scipy import stats
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestRegressor as RFR

def pear_corr(y_true, y_pred):
    """Calculate pearson correlation

    Parameters
    ----------
    y_true
    y_pred

    Returns
    -------
        r : float
            correlation value
    """
    r, _ = stats.pearsonr(y_true,y_pred)
    return r


def run_model(modelb, X_train_red, X_val_red, y_train, y_val):
    """run regression tranining
    
    """
    # Fit model.
    modelb.fit(X_train_red, y_train)

    # Model validation.
    y_val_pred = modelb.predict(X_val_red)
    val_result = pear_corr(y_val, y_val_pred)

    return modelb, val_result


def ftune_rf(cfg,  X_train_red, X_val_red, y_train, y_val):
    """Random-Forest FineTunner

    :param cfg: CFG
        configuration object.
    """
    model_list = []
    nb_jobs = 0

    for bootstrap in [ True, False]:
        for max_depth in [5, 7, 9]:
            for max_features in ['auto']:
                for min_samples_leaf in [3 , 5, 7]:
                    for min_samples_split in [3, 5, 7]:
                        for n_estimators in [100, 150, 200]:
                            rf_model = RFR(bootstrap = bootstrap, max_depth=max_depth,\
                                        max_features = max_features, min_samples_leaf= min_samples_leaf,\
                                        min_samples_split = min_samples_split, n_estimators=n_estimators,\
                                        random_state = cfg.MODEL.SEED)
                            nb_jobs += 1
                            model_list.append(copy.deepcopy(rf_model))

    rf_results = joblib.Parallel(n_jobs=nb_jobs, pre_dispatch='all')(joblib.delayed(run_model)(copy.deepcopy(model_),  copy.deepcopy(X_train_red),
                                                            copy.deepcopy(X_val_red), copy.deepcopy(y_train), copy.deepcopy(y_val)) for model_ in model_list)
    rf_opt_index = np.argmax([x[1] for x in rf_results])
    rf_best_model = rf_results[rf_opt_index][0]
    rf_best_val_score = rf_results[rf_opt_index][1]

    return rf_best_model, rf_best_val_score


def ftune_rf_cv(cfg, X_train_red, X_val_red, y_train, y_val):
    """Random-Forest Cross-Val FineTunner

    :param modelb: 
        model base

    :param cfg: CFG
        configuration object.
    """
    cfg.REGRESSOR.RF.BOOTSTRAP = [ True, False]
    cfg.REGRESSOR.RF.MAX_DEPTH = [7, 9]
    cfg.REGRESSOR.RF.MAX_FEATURES = ['auto']
    cfg.REGRESSOR.RF.MIN_SAMPLES_LEAF = [2]
    cfg.REGRESSOR.RF.MIN_SAMPLES_SPLIT = [2]
    cfg.REGRESSOR.RF.N_ESTIMATORS = [150, 230]
    cfg.FINE_TUNING.N_JOBS = 8

    # cross-val model training.
    score_type = {'PEARSON': make_scorer(pear_corr, greater_is_better=True)}

    tuning_params = {'bootstrap': cfg.REGRESSOR.RF.BOOTSTRAP,
                    'max_depth': cfg.REGRESSOR.RF.MAX_DEPTH,
                    'max_features': cfg.REGRESSOR.RF.MAX_FEATURES,
                    'min_samples_leaf': cfg.REGRESSOR.RF.MIN_SAMPLES_LEAF,
                    'min_samples_split': cfg.REGRESSOR.RF.MIN_SAMPLES_SPLIT,
                    'n_estimators': cfg.REGRESSOR.RF.N_ESTIMATORS,
                    'random_state': [cfg.MODEL.SEED]}

    modelb = GridSearchCV(RFR(), tuning_params, cv= cfg.FINE_TUNING.CV,
                            scoring=score_type, refit=list(score_type.keys())[0],
                            n_jobs= cfg.FINE_TUNING.N_JOBS)

    modelb.fit(X_train_red, y_train)
    y_val_pred = modelb.predict(X_val_red)
    val_result = pear_corr(y_val, y_val_pred)

    return modelb, val_result