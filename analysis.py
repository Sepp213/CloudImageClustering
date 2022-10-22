import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import glob


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
        tmp_freq_smooth = savgol_filter(tmp_freq, 10, 2)                         # Higher values for the second entry --> smoother, but too high --> origininal shape gets lost
        distributions.append(tmp_freq_smooth)
    plt.figure(figsize=(10,5), layout='constrained')
    for cluster, distributions in zip(range(8), distributions):
        plt.plot(x, distributions, label='cluster ' + str(cluster))
    plt.title('k_patches = 100, patchsize = 32, steps_patches = 32')
    plt.ylim([0, 50])
    plt.xlabel('Cloud Fraction')
    plt.ylabel('Frequency')
    plt.legend()
    # plt.savefig(name + '_smooth.png')
    plt.show()


def plot_all(path):
    filenames = glob.glob(path + "*.csv")
    filenames.sort()
    fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(20,7.5))
    plt.subplots_adjust(wspace=0.075, hspace=0.15,left=0.035,top=0.99,right=0.99,bottom=0.115)
    x_counter = 0
    y_counter = 0
    for name in filenames[9:18]:
        df = pd.read_csv(name)
        df.rename(columns={"open sky max": "osmax", "open sky mean": "osmean", "max length scale": "maxls", "mean length scale": "meanls"},inplace=True)
        cluster_indices = []
        for cluster in range(8):
            tmp = np.where(df.cluster == cluster)
            cluster_indices.append(tmp[0])
        cluster_fractions = []
        for cluster in cluster_indices:
            #tmp = df.fraction[cluster]
            tmp = df.iOrg[cluster]
            cluster_fractions.append(tmp)
        distributions = []
        x = np.linspace(0, 1, num=101, endpoint=True)
        # x = np.linspace(0, 128, num=129, endpoint=True)
        for cluster in cluster_fractions:
            tmp_freq, tmp_bins = np.histogram(cluster, bins=101, range=[0,1])
            # tmp_freq, tmp_bins = np.histogram(cluster, bins=129, range=[0,128])
            tmp_freq_smooth = savgol_filter(tmp_freq, 3, 2)                         # Higher values for the second entry --> smoother, but too high --> origininal shape gets lost
            distributions.append(tmp_freq_smooth)
        for cluster, distributions in zip(range(8), distributions):
            if x_counter == y_counter == 0:
                ax[x_counter][y_counter].plot(x, distributions, label='cluster ' + str(cluster))
            else:
                ax[x_counter][y_counter].plot(x, distributions)
            plt.ylim(0,35)
            plt.xlim(0,1)
        if y_counter == 2:
            y_counter = 0
            x_counter += 1
        else:
            y_counter += 1
    for a in ax:
        for b in a:
            b.set(xlabel='iOrg', ylabel='Frequency')
            b.label_outer()
    fig.legend(loc='lower center', bbox_to_anchor=(0.035,0.005,0.955,0.15), ncol=8, borderaxespad=0, mode='expand')
    # fig.legend(loc='center', ncol=8)
    plt.savefig('k500_iOrg.eps')
    plt.show()


def nof(path):
    nof = np.zeros(9)
    cells = np.zeros((3,3))
    filenames = glob.glob(path + "*.csv")
    filenames.sort()
    counter = 0
    cells_row_counter = 0
    cells_col_counter = 0
    for name in filenames[9:18]:
        df = pd.read_csv(name)
        df.rename(columns={"open sky max": "osmax", "open sky mean": "osmean", "max length scale": "maxls", "mean length scale": "meanls"},inplace=True)
        cluster_indices = []
        for cluster in range(8):
            tmp = np.where(df.cluster == cluster)
            cluster_indices.append(tmp[0])
        cluster_fractions = []
        for cluster in cluster_indices:
            #tmp = df.fraction[cluster]
            tmp = df.iOrg[cluster]
            cluster_fractions.append(tmp)
        distributions = []
        for cluster in cluster_fractions:
            tmp_freq, tmp_bins = np.histogram(cluster, bins=101, range=[0,1]) # for all metrics in [0,1]
            # tmp_freq, tmp_bins = np.histogram(cluster, bins=129, range=[0,128]) # for meanls
            distributions.append(tmp_freq)
        first = []
        second = []
        tmp_distr_freq = []
        for index in range(len(distributions[0])):
            for distr_index in range(len(distributions)):
                tmp_distr_freq.append(distributions[distr_index][index])
            tmp_distr_freq.sort(reverse=True)
            first.append(tmp_distr_freq[0])
            second.append(tmp_distr_freq[1])
            tmp_distr_freq = []
        for value_first, value_second in zip(first, second):
            nof[counter] += (value_first - value_second)
        cells[cells_row_counter][cells_col_counter] = "{:05.3f}".format(nof[counter]/1000)
        counter += 1
        if cells_col_counter == 2:
            cells_col_counter = 0
            cells_row_counter += 1
        else:
            cells_col_counter += 1
    rows = ['step size = [1,1,1]', 'step size = [2,4,16]', 'step size = [3,8,32]']
    columns = ['patchsize = 3', 'patchsize = 8', 'patchsize = 32']
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.axis('tight')
    ax.table(cellText=cells, colLabels=columns, rowLabels=rows, loc='center')
    fig.tight_layout()
    plt.savefig('k500_table_iOrg.eps')
    plt.show()

    
        

