import os
from os.path import dirname, abspath
import random
import numpy as np
from data.loader import data_loader
from situ.acronym import load_acronym
from .trainer import trainer
from exposure.exposure import community_expo
from regressor.features import build_features
from regressor.regression import test_regressor
from modeling.builder import regressor_builder, clusteror_builder
from clusteror.clustering import test_clusteror


class VISPEL(object):
    """
    Construct a end-to-end pip-line for
    VIsual Exposure Learning (VISPEL).

    """

    def __init__(self, situation, N=-1):
        self.N = N
        self.root = abspath(__file__).split('/lervup_algo/vispel')[0]
        self.situation = situation
        self.situ_encoding = load_acronym(situation)
        self.clusteror = None
        self.regressor = None
        self.feature_selector = None

    def load_cfg(self, cfg):
        """Load model's configuration and its data
        
        """
        self.cfg = cfg
        self.set_seeds()
        self.X_train, self.X_val, self.X_test, self.X_community, \
        self.gt_expos, self.vis_concepts, \
        self.detectors, self.opt_threds = data_loader(self.root, self.cfg, self.situation, self.N)


    def set_seeds(self):
        random.seed(self.cfg.MODEL.SEED)
        np.random.seed(self.cfg.MODEL.SEED)

    def train_vispel(self):
        """

        :return:
        """
        if self.cfg.OUTPUT.VERBOSE:
            print("#-------------------------------------------------#")
            print("# Train user's visual exposure predictor          ")
            print("#                  %s          " % self.situ_encoding)
            print("#-------------------------------------------------#")

        # Initiate training models
        clusteror = clusteror_builder(self.cfg)
        regressor = regressor_builder(self.cfg)

        # Train ...
        trained_clusteror, trained_regressor, feature_selector = trainer(self.situation, self.X_train, self.X_community, \
                                                                         self.gt_expos, clusteror, regressor,
                                                                         self.detectors, self.opt_threds, self.cfg)

        self.clusteror = trained_clusteror
        self.regressor = trained_regressor
        self.feature_selector = feature_selector


    def test_vispel(self, test_set, half_vis=False, load_half_vis = False):
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

        if hasattr(self, 'select_detectors'):
            self.sel_detectors = {} # selected detectors
            self.sel_opt_threds = {} # corresponding threholds

        if half_vis and not load_half_vis:
            tol_concepts = [] # total concepts
            sel_concepts = [] # selected concepts

            for obj, _ in self.detectors.items():
                tol_concepts.append(obj)

            while len(sel_concepts) <= int(len(tol_concepts) / 2):
                index = np.random.randint(len(tol_concepts))
                if tol_concepts[index] not in sel_concepts:
                    sel_concepts.append(tol_concepts[index])

            for obj in sel_concepts:
                self.sel_detectors[obj] = self.detectors[obj]
                if obj in self.opt_threds:
                    self.sel_opt_threds[obj] = self.opt_threds[obj]
        elif not half_vis:
            self.sel_detectors = self.detectors # all detectors
            self.sel_opt_threds = self.opt_threds


        test_expo_features = community_expo(test_set, self.cfg.SOLVER.F_TOP,\
                                            self.sel_detectors, self.sel_opt_threds, self.cfg.DETECTOR.LOAD,
                                            self.cfg)

        reg_test_features, gt_test_expos = build_features(self.clusteror, test_expo_features,
                                                          self.gt_expos, self.cfg)

        test_clusteror(self.situation, self.clusteror, test_expo_features, self.cfg)

        # Perform feature transform
        if self.cfg.PCA.STATE:
            X_test_rd = self.feature_selector.transform(reg_test_features)
            pca_variance = sum(self.feature_selector.explained_variance_ratio_)
        else:
            X_test_rd = reg_test_features
            pca_variance = 0

        corr_score = test_regressor(self.regressor, self.situation,
                                    X_test_rd, gt_test_expos, pca_variance, self.cfg)

        return corr_score