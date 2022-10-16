import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

def plot_from_csv(name):
    df  = pd.read_csv(name)
    df.rename(columns={"open sky max": "osmax", "open sky mean": "osmean", "max length scale": "maxls", "mean length scale": "meanls"},inplace=True)
    cluster_indices = []
    for cluster in range(8):
        tmp = np.where(df.cluster == cluster)
        cluster_indices.append(tmp[0])
    cluster_fractions = []
    for cluster in cluster_indices:
        tmp = df.fraction[cluster]
        cluster_fractions.append(tmp)
    distributions = []
    x = np.linspace(0, 1, num=101, endpoint=True)
    for cluster in cluster_fractions:
        tmp_freq, tmp_bins = np.histogram(cluster, bins=101, range=[0,1])
        tmp_freq_smooth = savgol_filter(tmp_freq, 50, 2)                         # Higher values for the second entry --> smoother, but too high --> origininal shape gets lost
        distributions.append(tmp_freq_smooth)
    plt.figure(figsize=(10,5), layout='constrained')
    for cluster, distributions in zip(range(8), distributions):
        plt.plot(x, distributions, label='cluster ' + str(cluster))
    plt.title('Distribution of Cloud Fractions per Cluster')
    plt.ylim([0, 20])
    plt.xlabel('Cloud Fraction')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()