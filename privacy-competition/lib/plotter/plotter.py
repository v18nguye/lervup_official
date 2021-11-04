"""


"""
import numpy as np
import matplotlib.pyplot as plt

def divide_data(x,y):
    """
    Divide data into positive and negative groups

    :return:
    """
    x_pca_pos = []
    y_pos = []
    x_pca_neg = []
    y_neg = []

    for k in range(y.shape[0]):

        if y[k] >= 0:
            x_pca_pos.append(x[k, :])
            y_pos.append(y[k])

        else:
            x_pca_neg.append(x[k, :])
            y_neg.append(y[k])

    x_pca_pos = np.asarray(x_pca_pos)
    x_pca_neg = np.asarray(x_pca_neg)
    y_pos = np.asarray(y_pos)
    y_neg = np.asarray(y_neg)

    return x_pca_pos, y_pos, x_pca_neg, y_neg


def pca_plot(situ_name, x_pca, y_gt, y_pred, pca_var, N = 10):
    """

    :param x_pca:
    :param y_pred:
    :param y_gt:
    :param N: number of plotted points
    :param pca_var:
    :return:
    """
    # SET UP
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # VISUALIZATION
    fig = plt.figure(figsize=(12,7))
    ax = fig.add_subplot(1, 2, 1)
    x_pca_pos, y_pos, x_pca_neg, y_neg = divide_data(x_pca, y_gt)
    g1 = (x_pca_pos, y_pos)
    g2 = (x_pca_neg, y_neg)
    data = (g1, g2)
    colors = ('green', 'red')

    for data, color in zip(data, colors):
        x, y = data
        ax.scatter(x[:,0], x[:,1], alpha=0.8, c=color, edgecolors='none', s=30)

        for i, txt in enumerate(y):
            ax.annotate("{:.2f}".format(txt), (x[i,0], x[i,1]))

    ax.set_xlabel('pca1')
    ax.set_ylabel('pca2')
    ax.set_title('Ground Truth')

    ax = fig.add_subplot(1, 2, 2)
    x_pca_pos, y_pos, x_pca_neg, y_neg = divide_data(x_pca, y_pred)
    g1 = (x_pca_pos, y_pos)
    g2 = (x_pca_neg, y_neg)
    data = (g1, g2)
    colors = ('green', 'red')

    for data, color in zip(data, colors):
        x, y = data
        ax.scatter(x[:,0], x[:,1], alpha=0.8, c=color, edgecolors='none', s=30)

        for i, txt in enumerate(y):
            ax.annotate("{:.2f}".format(txt), (x[i,0], x[i,1]))

    ax.set_xlabel('pca1')
    ax.set_ylabel('pca2')
    ax.set_title('Prediction')

    plt.title('PCA Variance ' + str(pca_var*100))
    fig.savefig(situ_name+'.jpg')