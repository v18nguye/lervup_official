import scipy.stats as stats

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


def kendall_corr(y_true, y_pred):
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
    r, _ = stats.kendalltau(y_true,y_pred)

    return r