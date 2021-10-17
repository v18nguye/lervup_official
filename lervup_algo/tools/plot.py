
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')

def plot_1():
    situs = ['BANK', 'IT', 'WAIT', 'ACCOM']
    FRCNN = np.asarray([42,42,65,46])
    FRCNN_Focal = np.asarray([5,7,2,2])

    MNET = np.asarray([24,42,65,45])
    MNET_Focal = np.asarray([2,3,1,2])


    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.set_ylabel('Correlation')
    ax1.bar(situs, FRCNN, label='F-RCNN')
    ax1.bar(situs, FRCNN_Focal, bottom=FRCNN, label='F-RCNN + FE')
    for xpos, ypos, yval in zip(situs, FRCNN+FRCNN_Focal, FRCNN_Focal):
        ax1.text(xpos, ypos, "%d"%yval+' %', ha="center", va="bottom")
    ax1.legend(loc = 2, fontsize = 'x-small')

    ax2.bar(situs, MNET, label='MNET')
    ax2.bar(situs, MNET_Focal, bottom=MNET, label='MNET+FE')
    for xpos, ypos, yval in zip(situs, MNET+MNET_Focal, MNET_Focal):
        ax2.text(xpos, ypos, "%d"%yval+' %', ha="center", va="bottom")

    ax2.legend(loc = 2, fontsize = 'x-small')
    fig.savefig('FE.png')

def plot_2():
    situs = ['BANK', 'IT', 'WAIT', 'ACCOM']


