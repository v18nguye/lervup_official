import os
import numpy as np
import scipy.stats as stats
from corr.corr_type import pear_corr, kendall_corr
from plotter.plotter import pca_plot
from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler, MaxAbsScaler

def train_regressor(model, x_train, y_train, cfg):
    """Train regressor by each situation

    :param: x_train: numpy array
        training data

    :param: y_train: numpy array
        training target

    :return:
        trained model

    : ref:
        https://scikit-learn.org/stable/modules/preprocessing.html
    """

    reg_feat_transform = RobustScaler().fit(x_train) # regression feature transformer
    x_train_transform = reg_feat_transform.transform(x_train)


    if cfg.REGRESSOR.TYPE == 'RF':
        model.fit(x_train_transform, y_train)

    if cfg.REGRESSOR.TYPE == 'SVM':
        model.fit(x_train_transform, y_train)

    if cfg.OUTPUT.VERBOSE and cfg.FINE_TUNING.STATUS:
        print('Best fine_tuned parameters: ')
        print(model.best_params_)


    y_pred = model.predict(x_train_transform)

    if cfg.OUTPUT.VERBOSE:

        if cfg.SOLVER.CORR_TYPE == 'KENDALL':
            print('correlation: ', "{:.4f}".format(kendall_corr(y_pred, y_train)))

        elif cfg.SOLVER.CORR_TYPE == 'PEARSON':
            print('correlation: ', "{:.4f}".format(pear_corr(y_pred, y_train)))

    return model, reg_feat_transform


def test_regressor(model, transformer, situ_name, x_test, y_test, cfg):
    """Train regressor by each situation

    :param transformer: 
        feature transformer (normalization, standarlize, etc ..)

    :param x_test: numpy array
        training data

    :param y_test: numpy array
        training target

    """
    x_test_transform = transformer.transform(x_test)

    if cfg.REGRESSOR.TYPE == 'RF':
        y_pred = model.predict(x_test_transform)

    if cfg.REGRESSOR.TYPE == 'SVM':
        y_pred = model.predict(x_test_transform)

    if cfg.SOLVER.CORR_TYPE == 'KENDALL':
        corr = kendall_corr(y_pred, y_test)
        if cfg.OUTPUT.VERBOSE:
            print('correlation: ', "{:.4f}".format(kendall_corr(y_pred, y_test)))

    elif cfg.SOLVER.CORR_TYPE == 'PEARSON':
        corr = pear_corr(y_pred, y_test)
        if cfg.OUTPUT.VERBOSE:
            print('correlation: ', "{:.4f}".format(pear_corr(y_pred, y_test)))

    return corr