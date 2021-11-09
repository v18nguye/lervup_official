import os
from os.path import dirname, abspath
import random
import numpy as np
from situ.acronym import load_acronym
from .trainer import trainer
from exposure.exposure import community_expo
from regressor.features import build_features
from regressor.regression import test_regressor
from modeling.builder import regressor_builder, clusteror_builder


class VISPEL(object):
    """
    Construct a end-to-end pip-line for
    VIsual Exposure Learning (VISPEL).

    """

    def __init__(self, situation):
        self.root = abspath(__file__).split('/lervup_algo/vispel')[0]
        self.situ_name = situation
        self.situ_encoding = load_acronym(self.situ_name)
        self.clus_feat_transform = None # clustering feature transform.
        self.reg_feat_transform = None # regression feature transform.
        self.feat_selector = None

    def init_model(self, cfg):
        """Initiate model training 

        """
        random.seed(cfg.MODEL.SEED)
        np.random.seed(cfg.MODEL.SEED)
        self.cfg = cfg
        self.clusteror = clusteror_builder(self.cfg)
        self.regressor = regressor_builder(self.cfg)

    def train_vispel(self, X_train, gt_expos, detectors, opt_threds):
        """Train vispel algo

        :param X_train: dict
            ...
        :param X_community:
            ...
        :param gt_expos:

        :return:
        """
        if self.cfg.OUTPUT.VERBOSE:
            print("#-------------------------------------------------#")
            print("# Train user's visual exposure predictor          ")
            print("#                  %s          " % self.situ_encoding)
            print("#-------------------------------------------------#")

        # Train ...
        clus_feat_transform, reg_feat_transform, feat_selector = trainer(self.situ_name, X_train, \
                                                                        gt_expos, self.clusteror, self.regressor, \
                                                                        detectors, opt_threds, self.cfg)

        self.clus_feat_transform = clus_feat_transform
        self.reg_feat_transform = reg_feat_transform
        self.feat_selector = feat_selector


    def test_vispel(self, X_valtest, gt_expos, detectors, opt_threds):
        """

        :param test_set: dict
            data to test a trained model

        :param half_vis: boolean
            take into account a half of number visual concepts
        :param load_half_vis : bloolean
            if load randomly pre-selected visual concepts
        """

        if self.cfg.OUTPUT.VERBOSE:
            print("#-------------------------------------------------#")
            print("# Evaluate user's visual exposure predictor       ")
            print("#-------------------------------------------------#")

        test_expo_features = community_expo(X_valtest, self.cfg.SOLVER.F_TOP,\
                                            detectors, opt_threds, self.cfg.DETECTOR.LOAD,
                                            self.cfg)

        reg_test_features, gt_test_expos = build_features(self.clusteror, self.clus_feat_transform, test_expo_features,
                                                          gt_expos, self.cfg)

        # Perform pca-based feature transform
        if self.cfg.PCA.APPLY:
            X_test_rd = self.feat_selector.transform(reg_test_features)
        else:
            X_test_rd = reg_test_features

        corr_score = test_regressor(self.regressor, self.reg_feat_transform, self.situ_name, \
                                    X_test_rd, gt_test_expos, self.cfg)

        return corr_score