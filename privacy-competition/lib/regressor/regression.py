import os
import numpy as np
import scipy.stats as stats
from corr.corr_type import pear_corr, kendall_corr
from plotter.plotter import pca_plot

def train_regressor(model, x_train, y_train, cfg):
    """Train regressor by each situation

    :param: x_train: numpy array
        training data

    :param: y_train: numpy array
        training target

    :return:
        trained modeling

    """

    if cfg.REGRESSOR.TYPE == 'RF':
        model.fit(x_train, y_train)

    if cfg.REGRESSOR.TYPE == 'SVM':
        model.fit(x_train, y_train)

    if cfg.OUTPUT.VERBOSE and cfg.FINE_TUNING.STATUS:
        print('Best fine_tuning parameters: ')
        print(model.best_params_)


    y_pred = model.predict(x_train)

    if cfg.OUTPUT.VERBOSE:

        if cfg.SOLVER.CORR_TYPE == 'KENDALL':
            print('correlation: ', kendall_corr(y_pred, y_train))

        elif cfg.SOLVER.CORR_TYPE == 'PEARSON':
            print('correlation: ', pear_corr(y_pred, y_train))

    return model


def test_regressor(model, situ_name, x_test, y_test, pca_var, cfg):
    """Train regressor by each situation

    :param: x_test: numpy array
        training data

    :param: y_test: numpy array
        training target

    :return:
        trained modeling

    """
    if cfg.REGRESSOR.TYPE == 'RF':
        y_pred = model.predict(x_test)

    if cfg.REGRESSOR.TYPE == 'SVM':
        y_pred = model.predict(x_test)

    if cfg.PCA.STATE:
        pca_plot(situ_name, x_test, y_test, y_pred, pca_var)

    if cfg.SOLVER.CORR_TYPE == 'KENDALL':
        corr = kendall_corr(y_pred, y_test)
        if cfg.OUTPUT.VERBOSE:
            print('correlation: ', kendall_corr(y_pred, y_test))

    elif cfg.SOLVER.CORR_TYPE == 'PEARSON':
        corr = pear_corr(y_pred, y_test)
        if cfg.OUTPUT.VERBOSE:
            print('correlation: ', pear_corr(y_pred, y_test))

    return corr


def test_regressor_v2(model, situ_name, x_test, y_test, pca_var, cfg):
    """Train regressor by each situation

    :param: x_test: numpy array
        training data

    :param: y_test: numpy array
        training target

    :return:
        trained modeling

    """
    if cfg.REGRESSOR.TYPE == 'RF':
        y_pred = model.predict(x_test)

    if cfg.REGRESSOR.TYPE == 'SVM':
        y_pred = model.predict(x_test)

    return y_pred