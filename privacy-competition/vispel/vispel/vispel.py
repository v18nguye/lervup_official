import os
import random
import numpy as np
import json
from data.loader import data_loader
from situ.acronym import load_acronym
from vispel.trainer import trainer
from exposure.exposure import community_expo
from regressor.features import build_features, build_test_feats
from regressor.regression import test_regressor, test_regressor_v2
from modeling.builder import regressor_builder, clusteror_builder
from clusteror.clustering import test_clusteror


class VISPEL(object):
    """
    Construct a end-to-end training pip-line for the VISPEL predictor

    """

    def __init__(self, cfg, situation, N=-1):
        self.cfg = cfg
        self.N = N
        self.root = os.getcwd().split('/privacy-competition/tools')[0]
        self.situation = situation
        self.situ_encoding = load_acronym(situation)
        self.X_train, self.X_test, self.X_community, \
        self.gt_expos, self.vis_concepts, \
        self.detectors, self.opt_threds = data_loader(self.root, self.cfg, self.situation, self.N)
        self.clusteror = None
        self.regressor = None
        self.feature_selector = None
        self.test_result = None
        self.set_seeds()

    def set_seeds(self):
        random.seed(self.cfg.MODEL.SEED)
        np.random.seed(self.cfg.MODEL.SEED)

    def train_vispel(self):
        """

        :return:
        """
        if self.cfg.OUTPUT.VERBOSE:
            print("#-------------------------------------------------#")
            print("# Train visual privacy exposure predictor          ")
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

    def test_vispel(self, half_vis=False):
        """
        half_vis: float
            take into account a half of number visual concepts
        """

        if self.cfg.OUTPUT.VERBOSE:
            print("#-------------------------------------------------#")
            print("# Evaluate visual privacy exposure predictor       ")
            print("#-------------------------------------------------#")

        sel_detectors = {}
        sel_opt_threds = {}

        if half_vis:
            tol_concepts = []
            sel_concepts = []

            for obj, _ in self.detectors.items():
                tol_concepts.append(obj)

            while len(sel_concepts) <= int(len(tol_concepts) / 2):
                index = np.random.randint(len(tol_concepts))
                if tol_concepts[index] not in sel_concepts:
                    sel_concepts.append(tol_concepts[index])

            for obj in sel_concepts:
                sel_detectors[obj] = self.detectors[obj]
                if obj in self.opt_threds:
                    sel_opt_threds[obj] = self.opt_threds[obj]
        else:
            sel_detectors = self.detectors
            sel_opt_threds = self.opt_threds

        test_expo_features = community_expo(self.X_test, self.cfg.SOLVER.F_TOP, \
                                            sel_detectors, sel_opt_threds, self.cfg.DETECTOR.LOAD,
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

        self.test_result = corr_score

    def inference(self):
        """Run inference on new images and return scores

        """
        sel_detectors = self.detectors
        sel_opt_threds = self.opt_threds
        X_test_new = json.load(
            open('/home/nguyen/Documents/intern20/Vis-Priva-Expos/privacy-competition/data/test_preds.json'))
        test_expo_features = community_expo(X_test_new, self.cfg.SOLVER.F_TOP,\
                                            sel_detectors, sel_opt_threds, self.cfg.DETECTOR.LOAD,
                                            self.cfg)

        reg_test_features, users = build_test_feats(self.clusteror, test_expo_features,
                                             self.gt_expos, self.cfg)

        X_test_rd = reg_test_features
        pca_variance = 0

        pred_scores = test_regressor_v2(self.regressor, self.situation,
                                     X_test_rd, None, pca_variance, self.cfg)

        pred_dict = {}

        for idx, user in enumerate(users):
            pred_dict[user] = pred_scores[idx]
            
        return pred_dict
