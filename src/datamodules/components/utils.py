import matplotlib.pyplot as plt
import numpy as np

def qplot(x):
    pctls = np.linspace(0,100,21)[1:-1]
    qtles = np.percentile(x, pctls)
    plt.plot(pctls, qtles)
    plt.grid(); plt.xlabel("percentiles"); plt.ylabel("quantiles")

def pimshow(x, pmin=1, pmax=99, set_nans_to=0, alpha=None):
    xx = x.copy()
    xx[np.isnan(xx)]=set_nans_to
    vmin, vmax = np.percentile(xx, [pmin, pmax])
    plt.imshow(x, vmin=vmin, vmax=vmax, alpha=alpha)