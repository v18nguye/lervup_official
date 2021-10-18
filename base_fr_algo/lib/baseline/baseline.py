import os
import numpy as np
import random
from loader.loader import bloader
from situ.acronym import situ_decoding
from optimal_search.correlation import corr
from optimal_search.max_tau_subset import tau_subset, tau_max_cross_val
from optimal_search.optimal_thres_object import search_optimal_thres

class BaseFocalRating(object):
    """
    Construct the user exposure prediction baseline
    with focal rating.

    """

    def __init__(self, situ, save_file):
        self.opt_thresholds = {}  # optimal threshold for each visual concept
        self.opt_detectors = {}  # selected detectors within its optimal thresholds
        self.save_file = save_file
        self.situ = situ_decoding(situ)
        self.root = os.getcwd().split('/base_fr_algo')[0]

    def load_cfg(self, cfg):
        """Load model's configuration and its data
        
        """
        self.cfg = cfg
        self.set_seeds()
        self.save_path = os.path.join(self.cfg.OUTPUT.DIR, self.save_file.split('.pkl')[0] + '.txt')
        self.x_train, self.x_val, self.x_test, self.detectors, self.gt_expos = bloader(self.root, self.cfg, self.situ)

    def set_seeds(self):
        random.seed(self.cfg.MODEL.SEED)
        np.random.seed(self.cfg.MODEL.SEED)

    def train(self):
        # optimal threshold for each vis concept
        self.opt_thresholds = search_optimal_thres(self.x_train, self.gt_expos, self.detectors,
                                                   self.cfg.SOLVER.CORR_TYPE, self.cfg)

        # optimal subset of vis concepts for each situ
        if self.cfg.SOLVER.CROSS_VAL:
            _, _, opt_detectors = tau_max_cross_val(self.x_train, self.gt_expos,
                                                                self.opt_thresholds, self.cfg.SOLVER.CORR_TYPE,
                                                                self.cfg, self.cfg.SOLVER.K_FOLDS)
            self.opt_detectors = opt_detectors

        else:
            _, opt_detectors, _, _ = tau_subset(self.x_train, self.gt_expos,
                                                    self.opt_thresholds, self.cfg.SOLVER.CORR_TYPE, self.cfg)
            self.opt_detectors = opt_detectors

    def test(self, test_set):
        result = corr(test_set, self.gt_expos,
                            self.opt_detectors, self.cfg.SOLVER.CORR_TYPE, self.cfg, test_mode=True)
        return result