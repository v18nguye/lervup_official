# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .config import CfgNode as VISPEL

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = VISPEL()
# The version number, to upgrade from old configs to new ones if any
# changes happen. It's recommended to keep a VERSION in your config file.
_C.VERSION = 1
_C.MODEL = VISPEL()
# Setting in the debug mode
_C.MODEL.DEBUG = False
# Setting in seed for models
_C.MODEL.SEED = 2021
# Architecture meta-data
_C.MODEL.META_ARCHITECTURE = "VISual-Privacy-Exposure-Learner"

# ---------------------------------------------------------------------------- #
# OUTPUT
# ---------------------------------------------------------------------------- #
_C.OUTPUT = VISPEL()
_C.OUTPUT.DIR = ''
_C.OUTPUT.VERBOSE = False

# ---------------------------------------------------------------------------- #
# Grid Search
# ---------------------------------------------------------------------------- #
_C.GRID_SEARCH = VISPEL()
# Fine-tune modeling parameters
_C.GRID_SEARCH.STATUS = False
# Cross validation
_C.GRID_SEARCH.CV = 10
# Number of used jobs
_C.GRID_SEARCH.N_JOBS = -1

# ---------------------------------------------------------------------------- #
# FEATURE REDUCTION - PCA
# ---------------------------------------------------------------------------- #
_C.PCA = VISPEL()
# if apply the feature reduction.
_C.PCA.APPLY = False
# Number of components to keep after
# performing feature reduction.
# If None, use all components.
_C.PCA.N_COMPONENTS = 2

# -----------------------------------------------------------------------------
# DETECTOR
# -----------------------------------------------------------------------------
_C.DETECTOR = VISPEL()
# Load pre-defined detectors which was determined in
# the privacy baseline algorithm.
_C.DETECTOR.LOAD = False

# -----------------------------------------------------------------------------
# SELECT good training users
# -----------------------------------------------------------------------------
_C.USER_SELECTOR = VISPEL()
# If apply the user selection process
_C.USER_SELECTOR.STATE = False
# Absolute visual score difference between two users. If two
# users have the score difference smaller than EPS, they will
# take into account to calculate the feature distance
_C.USER_SELECTOR.EPS = 0.05
# Percentage of kept users
# following their ranked feature distance
_C.USER_SELECTOR.KEEP = 1.0

# -----------------------------------------------------------------------------
# DATASETS
# -----------------------------------------------------------------------------
_C.DATASETS = VISPEL()
# Train, test splits of user profiles.
_C.DATASETS.TRAIN_TEST_SPLIT = ''
# Crowd-sourcing user exposure corr.
_C.DATASETS.GT_USER_EXPOS = ''
# Crowd-sourcing visual concepts in different situations
_C.DATASETS.VIS_CONCEPTS = ''
# Pre-selected visual concepts in different situations,
# given by the privacy base-line
_C.DATASETS.PRE_VIS_CONCEPTS = ''

# ---------------------------------------------------------------------------- #
# SOLVER
# ---------------------------------------------------------------------------- #
_C.SOLVER = VISPEL()
# Top object detection confidence scores of a detector
# in a considered image.
_C.SOLVER.F_TOP = 0.2
# Feature transform applied in photos. These types include: ORG, VOTE
# - ORG: Original Features
# - VOTE: Select the high feature exposure
_C.SOLVER.FEATURE_TYPE = 'ORG'
# Currently supported correlation types: KENDALL, PEARSON
# Evaluate the correlation score between the crowd-sourcing user exposure corr
# and the learned user exposure corr.
_C.SOLVER.CORR_TYPE = 'KENDALL'
# Filtering neutral images whose absolute exposure sum is smaller than 0.01. The
# accepted images should satisfy the following condition:
#               abs(negative_scaled_exposure) + positive_scaled_exposure > FILT_THRESHOLD
_C.SOLVER.FILTERING = True
# FILTERING THRESHOLD
_C.SOLVER.FILT_THRESHOLD = 0.01

# ---------------------------------------------------------------------------- #
# FOCAL EXPOSURE \ FOCAL RATING
# ---------------------------------------------------------------------------- #
_C.FE = VISPEL()
# Focusing factor in the Focal Exposure (FE) function.
_C.FE.GAMMA = 2
# Scaling constant in the Focal Exposure (FE) function.
_C.FE.K = 4
# Extreme visual concept score lower threshold.
_C.FE.TAU_e = 1
# Ratio threshold between extreme and total concepts
_C.FE.TAU_o = 1/3
# How to apply Focal Exposure
# + IMAGE: apply in the image level
# + OBJECT: apply in the object level
_C.FE.MODE = 'IMAGE'

# ---------------------------------------------------------------------------- #
# CLUSTEROR
# ---------------------------------------------------------------------------- #
_C.CLUSTEROR = VISPEL()
# Currently supported clustering algorithm(s):
# - k-means (K-MEANS)
# - gaussian mixture modeling (GM)
_C.CLUSTEROR.TYPE = 'K_MEANS'

# ---------------------------------------------------------------------------- #
# K MEANS
# ---------------------------------------------------------------------------- #
_C.CLUSTEROR.K_MEANS = VISPEL()
# Number of pre-defined clusters in the K-means algorithm.
_C.CLUSTEROR.K_MEANS.CLUSTERS = 4
_C.CLUSTEROR.K_MEANS.N_INIT = 10
_C.CLUSTEROR.K_MEANS.MAX_ITER = 500
_C.CLUSTEROR.K_MEANS.ALGORITHM = 'auto'

# ---------------------------------------------------------------------------- #
# GAUSSIAN MIXTURE MODELS
# ---------------------------------------------------------------------------- #
_C.CLUSTEROR.GM = VISPEL()
_C.CLUSTEROR.GM.COMPONENTS = 4
_C.CLUSTEROR.GM.MAX_ITER = 100
_C.CLUSTEROR.GM.COV_TYPE = 'full'

# ---------------------------------------------------------------------------- #
# REGRESSOR
# ---------------------------------------------------------------------------- #
_C.REGRESSOR = VISPEL()
# Currently supported learning algorithms:
# - random forest (RF)
# - support vector machine (SVM).
_C.REGRESSOR.TYPE = 'RF'
# If use centroids (CENTROIDS) given by clustering all images in the training community
# as the features for each user's exposure. If not, the user's centroids (MEANS) calculated
# by averaging exposures on selected exposure features in each cluster will
# be taken to replace the centroids.
# Regression features, currently supported types:
# - FR1: CENTROIDS + VARIANCE ( K-MEANS, GM)
# - FR2: MEANS + VARIANCE ( K-MEANS, GM)
_C.REGRESSOR.FEATURES = 'FR2'

# ---------------------------------------------------------------------------- #
# SUPPORT VECTOR MACHINE
# ---------------------------------------------------------------------------- #
_C.REGRESSOR.SVM = VISPEL()
_C.REGRESSOR.SVM.KERNEL = ['rbf']
_C.REGRESSOR.SVM.GAMMA = [1e-3]
_C.REGRESSOR.SVM.C = [5]

# ---------------------------------------------------------------------------- #
# RANDOM FOREST
# ---------------------------------------------------------------------------- #
_C.REGRESSOR.RF = VISPEL()
_C.REGRESSOR.RF.BOOTSTRAP = [True]
_C.REGRESSOR.RF.MAX_DEPTH = [7]
_C.REGRESSOR.RF.MAX_FEATURES = ['auto']
_C.REGRESSOR.RF.MIN_SAMPLES_LEAF = [1, 3, 5]
_C.REGRESSOR.RF.MIN_SAMPLES_SPLIT = [2, 4, 6]
_C.REGRESSOR.RF.N_ESTIMATORS = [150, 200]