import math
import numpy as np

def cut_off(x, tau = 9):

    if abs(x) <= tau:
        x_cut = x
    else:
        x_cut = np.sign(x)*9

    return x_cut


def focal_exposure(expo, gamma, K = 10):
    """Rescale  photo exposures

    Parameters
    ----------
        expo : float
            orginial photo exposure
        
        gamma : int
            focusing factor

        K : float
            rescaling constant

    Returns
    -------
        rescaled photo exposure
    """
    scaled_expo =  (1/(1-(1/K)*abs(expo))**gamma)*expo

    return cut_off(scaled_expo)