import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import glob
import matplotlib
import random as rand
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid



def boxplots_cloud_fraction(csv_path):
    df = pd.read_csv(csv_path)
    df.rename(columns={"open sky max": "osmax", "open sky mean": "osmean", "max length scale": "maxls", "mean length scale": "meanls"},inplace=True)
    df['fraction'] = df['fraction'].fillna(0)
    cluster_indices = []
    for cluster in range(7):
        tmp = np.where(df.cluster == cluster + 1)
        cluster_indices.append(tmp[0])
    cloud_fractions = []
    for cluster in cluster_indices:
        tmp = df.fraction[cluster]
        cloud_fractions.append(tmp)
    fig = plt.figure(figsize =(10, 7))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(cloud_fractions, patch_artist = True, notch ='True', vert = 0, showfliers=False)

    # colors = []
    # for patch, color in zip(bp['boxes'], colors):
    #     patch.set_facecolor(color)
 
    # changing color and linewidth of
    # caps
    for cap in bp['caps']:
        cap.set(color ='#8B008B', linewidth = 2)
    
    # changing color and linewidth of
    # medians
    for median in bp['medians']:
        median.set(color ='red', linewidth = 3)
    
    # changing style of fliers
    for flier in bp['fliers']:
        flier.set(marker ='D', color ='#e7298a', alpha = 0.5)
        
    # x-axis labels
    ax.set_yticklabels(['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Cluster 6', 'Cluster 7'])
    
    # Adding title
    plt.title("Cloud Fractions per Cluster")
    
    # Removing top axes and right axes
    # ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    x_grid_points = np.linspace(0,1,num=21)
    ax.xaxis.set_ticks(x_grid_points)
    # xticks = ax.xaxis.get_major_ticks()
    # counter = 0
    plt.grid()
    # for tick in xticks:
    #     if counter % 5 != 0:
    #         tick.set_visible(False)
    #     counter += 1
    plt.xlabel('Cloud Fraction')
    plt.tight_layout()
    plt.savefig('boxplots_cloud_fraction.eps')
    plt.show()


def boxplots_osmean(csv_path):
    df = pd.read_csv(csv_path)
    df.rename(columns={"open sky max": "osmax", "open sky mean": "osmean", "max length scale": "maxls", "mean length scale": "meanls"},inplace=True)
    df['osmean'] = df['osmean'].fillna(0)
    cluster_indices = []
    for cluster in range(7):
        tmp = np.where(df.cluster == cluster + 1)
        cluster_indices.append(tmp[0])
    osmeans = []
    for cluster in cluster_indices:
        tmp = df.osmean[cluster]
        osmeans.append(tmp)
    fig = plt.figure(figsize =(10, 7))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(osmeans, patch_artist = True, notch ='True', vert = 0, showfliers=False)

    # colors = []
    # for patch, color in zip(bp['boxes'], colors):
    #     patch.set_facecolor(color)
 
    # changing color and linewidth of
    # caps
    for cap in bp['caps']:
        cap.set(color ='#8B008B', linewidth = 2)
    
    # changing color and linewidth of
    # medians
    for median in bp['medians']:
        median.set(color ='red', linewidth = 3)
    
    # changing style of fliers
    for flier in bp['fliers']:
        flier.set(marker ='D', color ='#e7298a', alpha = 0.5)
        
    # x-axis labels
    ax.set_yticklabels(['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Cluster 6', 'Cluster 7'])
    
    # Adding title
    plt.title("Open Sky Mean per Cluster")
    
    # Removing top axes and right axes
    # ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    x_grid_points = np.linspace(0,1,num=21)
    ax.xaxis.set_ticks(x_grid_points)
    # xticks = ax.xaxis.get_major_ticks()
    # counter = 0
    plt.grid()
    # for tick in xticks:
    #     if counter % 5 != 0:
    #         tick.set_visible(False)
    #     counter += 1
    plt.xlabel('Open Sky Mean')
    plt.tight_layout()
    plt.savefig('boxplots_osmean.eps')
    plt.show()


def boxplots_meanls(csv_path):
    df = pd.read_csv(csv_path)
    df.rename(columns={"open sky max": "osmax", "open sky mean": "osmean", "max length scale": "maxls", "mean length scale": "meanls"},inplace=True)
    df['meanls'] = df['meanls'].fillna(0)
    cluster_indices = []
    for cluster in range(7):
        tmp = np.where(df.cluster == cluster + 1)
        cluster_indices.append(tmp[0])
    meanls = []
    for cluster in cluster_indices:
        tmp = df.meanls[cluster]
        meanls.append(tmp)
    fig = plt.figure(figsize =(10, 7))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(meanls, patch_artist = True, notch ='True', vert = 0, showfliers=False)

    # colors = []
    # for patch, color in zip(bp['boxes'], colors):
    #     patch.set_facecolor(color)
 
    # changing color and linewidth of
    # caps
    for cap in bp['caps']:
        cap.set(color ='#8B008B', linewidth = 2)
    
    # changing color and linewidth of
    # medians
    for median in bp['medians']:
        median.set(color ='red', linewidth = 3)
    
    # changing style of fliers
    for flier in bp['fliers']:
        flier.set(marker ='D', color ='#e7298a', alpha = 0.5)
        
    # x-axis labels
    ax.set_yticklabels(['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Cluster 6', 'Cluster 7'])
    
    # Adding title
    plt.title("Mean Length Scale per Cluster")
    
    # Removing top axes and right axes
    # ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    x_grid_points = np.linspace(0,128,num=17)
    ax.xaxis.set_ticks(x_grid_points)
    # xticks = ax.xaxis.get_major_ticks()
    # counter = 0
    plt.grid()
    # for tick in xticks:
    #     if counter % 5 != 0:
    #         tick.set_visible(False)
    #     counter += 1
    plt.xlabel('Mean Length Scale')
    plt.tight_layout()
    plt.savefig('boxplots_meanls.eps')
    plt.show()


def boxplots_orientation(csv_path):
    df = pd.read_csv(csv_path)
    df.rename(columns={"open sky max": "osmax", "open sky mean": "osmean", "max length scale": "maxls", "mean length scale": "meanls"},inplace=True)
    df['orientation'] = df['orientation'].fillna(0)
    cluster_indices = []
    for cluster in range(7):
        tmp = np.where(df.cluster == cluster + 1)
        cluster_indices.append(tmp[0])
    orientations = []
    for cluster in cluster_indices:
        tmp = df.orientation[cluster]
        orientations.append(tmp)
    fig = plt.figure(figsize =(10, 7))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(orientations, patch_artist = True, notch ='True', vert = 0, showfliers=False)

    # colors = []
    # for patch, color in zip(bp['boxes'], colors):
    #     patch.set_facecolor(color)
 
    # changing color and linewidth of
    # caps
    for cap in bp['caps']:
        cap.set(color ='#8B008B', linewidth = 2)
    
    # changing color and linewidth of
    # medians
    for median in bp['medians']:
        median.set(color ='red', linewidth = 3)
    
    # changing style of fliers
    for flier in bp['fliers']:
        flier.set(marker ='D', color ='#e7298a', alpha = 0.5)
        
    # x-axis labels
    ax.set_yticklabels(['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Cluster 6', 'Cluster 7'])
    
    # Adding title
    plt.title("Orientation per Cluster")
    
    # Removing top axes and right axes
    # ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    x_grid_points = np.linspace(0,1,num=21)
    ax.xaxis.set_ticks(x_grid_points)
    # xticks = ax.xaxis.get_major_ticks()
    # counter = 0
    plt.grid()
    # for tick in xticks:
    #     if counter % 5 != 0:
    #         tick.set_visible(False)
    #     counter += 1
    plt.xlabel('Orientation')
    plt.tight_layout()
    plt.savefig('boxplots_orientation.eps')
    plt.show()


def boxplots_iorg(csv_path):
    df = pd.read_csv(csv_path)
    df.rename(columns={"open sky max": "osmax", "open sky mean": "osmean", "max length scale": "maxls", "mean length scale": "meanls"},inplace=True)
    df['iOrg'] = df['iOrg'].fillna(0)
    cluster_indices = []
    for cluster in range(7):
        tmp = np.where(df.cluster == cluster + 1)
        cluster_indices.append(tmp[0])
    iOrgs = []
    for cluster in cluster_indices:
        tmp = df.iOrg[cluster]
        iOrgs.append(tmp)
    fig = plt.figure(figsize =(10, 7))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(iOrgs, patch_artist = True, notch ='True', vert = 0, showfliers=False)

    # colors = []
    # for patch, color in zip(bp['boxes'], colors):
    #     patch.set_facecolor(color)
 
    # changing color and linewidth of
    # caps
    for cap in bp['caps']:
        cap.set(color ='#8B008B', linewidth = 2)
    
    # changing color and linewidth of
    # medians
    for median in bp['medians']:
        median.set(color ='red', linewidth = 3)
    
    # changing style of fliers
    for flier in bp['fliers']:
        flier.set(marker ='D', color ='#e7298a', alpha = 0.5)
        
    # x-axis labels
    ax.set_yticklabels(['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Cluster 6', 'Cluster 7'])
    
    # Adding title
    plt.title("iOrg per Cluster")
    
    # Removing top axes and right axes
    # ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    x_grid_points = np.linspace(0,1,num=21)
    ax.xaxis.set_ticks(x_grid_points)
    # xticks = ax.xaxis.get_major_ticks()
    # counter = 0
    plt.grid()
    # for tick in xticks:
    #     if counter % 5 != 0:
    #         tick.set_visible(False)
    #     counter += 1
    plt.xlabel('iOrg')
    plt.tight_layout()
    plt.savefig('boxplots_iOrg.eps')
    plt.show()


def boxplots_cod(csv_path):
    df = pd.read_csv(csv_path)
    df.rename(columns={"open sky max": "osmax", "open sky mean": "osmean", "max length scale": "maxls", "mean length scale": "meanls"},inplace=True)
    df['cod'] = df['cod'].fillna(0)
    cluster_indices = []
    for cluster in range(7):
        tmp = np.where(df.cluster == cluster + 1)
        cluster_indices.append(tmp[0])
    cod = []
    for cluster in cluster_indices:
        tmp = df.cod[cluster]
        cod.append(tmp)
    fig = plt.figure(figsize =(10, 7))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(cod, patch_artist = True, notch ='True', vert = 0, showfliers=False)

    # colors = []
    # for patch, color in zip(bp['boxes'], colors):
    #     patch.set_facecolor(color)
 
    # changing color and linewidth of
    # caps
    for cap in bp['caps']:
        cap.set(color ='#8B008B', linewidth = 2)
    
    # changing color and linewidth of
    # medians
    for median in bp['medians']:
        median.set(color ='red', linewidth = 3)
    
    # changing style of fliers
    for flier in bp['fliers']:
        flier.set(marker ='D', color ='#e7298a', alpha = 0.5)
        
    # x-axis labels
    ax.set_yticklabels(['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Cluster 6', 'Cluster 7'])
    
    # Adding title
    plt.title("Image Mean Cloud Optical Depth per Cluster")
    
    # Removing top axes and right axes
    # ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    x_grid_points = np.linspace(0,70,num=15)
    ax.xaxis.set_ticks(x_grid_points)
    # xticks = ax.xaxis.get_major_ticks()
    # counter = 0
    plt.grid()
    # for tick in xticks:
    #     if counter % 5 != 0:
    #         tick.set_visible(False)
    #     counter += 1
    plt.xlabel('Mean Cloud Optical Depth')
    plt.tight_layout()
    plt.savefig('boxplots_cod.eps')
    plt.show()


def plot_from_csv(name):
    df  = pd.read_csv(name)
    df.rename(columns={"open sky max": "osmax", "open sky mean": "osmean", "max length scale": "maxls", "mean length scale": "meanls"},inplace=True)
    cluster_indices = []
    for cluster in range(8):
        tmp = np.where(df.cluster == cluster + 1)
        cluster_indices.append(tmp[0])
    cluster_fractions = []
    for cluster in cluster_indices:
        tmp = df.fraction[cluster]
        cluster_fractions.append(tmp)
    distributions = []
    x = np.linspace(0, 1, num=101, endpoint=True)
    for cluster in cluster_fractions:
        tmp_freq, tmp_bins = np.histogram(cluster, bins=101, range=[0,1])
        tmp_freq_smooth = savgol_filter(tmp_freq, 3, 2)                         # Higher values for the second entry --> smoother, but too high --> origininal shape gets lost
        distributions.append(tmp_freq_smooth)
    plt.figure(figsize=(10,5), layout='constrained')
    for cluster, distributions in zip(range(8), distributions):
        plt.plot(x, distributions, label='cluster ' + str(cluster))
    plt.title('k_patches = 100, patchsize = 8, steps_patches = 1')
    plt.ylim([0, 50])
    plt.xlabel('orientation')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('COD_orientation.png')
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


#------Plot Clustering------------------------------------------------------------------------------------------------------
def plot_clustering_csv(path, cmap, image_type):
    print('----------------------------------------------------------------------------------------------------')
    print('Creating plot.')
    df  = pd.read_csv(path)
    rows = 14                                                                                        
    columns = 20
    fig = plt.figure(figsize=(columns * 32, rows * 32))                                           
    row_counter = 0
    column_counter = 0
    filenames = df.path
    for cluster in range(rows):
        tmp = np.where(df.cluster == cluster + 1)
        tmp_list = tmp[0].tolist()
        sample_list = rand.sample(tmp_list, min(columns, len(tmp_list)))                                      
        for sample_counter in range(len(sample_list)):
            ax = fig.add_subplot(rows, columns, row_counter * columns + column_counter + 1)
            if image_type == 'cod':
                plt.imshow(Image.open(filenames[sample_list[sample_counter]]),cmap=cmap)
            elif image_type == 'cm':  
                plt.imshow(np.load(filenames[sample_list[sample_counter]]),cmap=cmap, vmin=0, vmax=1)                          
            plt.axis('off')
            column_counter += 1
        row_counter += 1
        column_counter = 0
    plt.subplots_adjust(0,0,1,1,0.1,0.1)
    plt.savefig("TEST_CSV.png", dpi=1)
    print('Plot saved.')
    print('----------------------------------------------------------------------------------------------------')
    print()


#------Plot Clustering------------------------------------------------------------------------------------------------------
def plot_clustering_csv2(path, cmap, image_type):
    print('----------------------------------------------------------------------------------------------------')
    print('Creating plot.')
    df  = pd.read_csv(path)
    rows = 14                                                                                        
    columns = 20
    fig = plt.figure(figsize=(columns * 32, rows * 32))                                           
    row_counter = 0
    column_counter = 0
    filenames = df.path
    images = []
    for i in filenames: 
        img_tmp = Image.open(i)
        images.append(np.log(np.array(img_tmp) + 1)) 
        img_tmp.close()   
    glob_max = []
    glob_min = []
    for img in images:
        glob_max.append(np.amax(img))
        glob_min.append(np.amin(img))
    images_norm2_reduced = []
    for img in images:
        images_norm2_reduced.append(np.divide(img, max(glob_max), dtype=float))
    for cluster in range(rows):
        tmp = np.where(df.cluster == cluster + 1)
        tmp_list = tmp[0].tolist()
        sample_list = rand.sample(tmp_list, min(columns, len(tmp_list)))                                      
        for sample_counter in range(len(sample_list)):
            ax = fig.add_subplot(rows, columns, row_counter * columns + column_counter + 1)
            plt.imshow(images_norm2_reduced[sample_list[sample_counter]],cmap=cmap, vmin=0, vmax=1)                          
            plt.axis('off')
            column_counter += 1
        row_counter += 1
        column_counter = 0
    plt.subplots_adjust(0,0,1,1,0.1,0.1)
    plt.savefig("TEST_CSV.png")
    print('Plot saved.')
    print('----------------------------------------------------------------------------------------------------')
    print()


def plot_images_treshs(images, images_norm):
    print('----------------------------------------------------------------------------------------------------')
    print('Creating plot.')
    rows = len(images)                                                                                        
    treshs = [0.5, 0.1, 0.01, 0.001]
    columns = len(treshs) + 2
    row_counter = 0     
    images_treshs = []
    for image, image_norm in zip(images, images_norm):
        images_treshs.append(image)
        images_treshs.append(image_norm)
        for tresh in treshs:
            images_treshs.append(np.where(image > tresh, 0, 1))
    fig = plt.figure(figsize=(rows * 16, columns * 16))
    grid = ImageGrid(fig, 111, nrows_ncols=(rows, columns), axes_pad=0.1)
    for img, ax in zip(images_treshs, grid):
        ax.imshow(img, cmap='Spectral')
        plt.axis('off')
    plt.savefig("TEST_TRESHS.png")
    print('Plot saved.')
    print('----------------------------------------------------------------------------------------------------')
    print()

    
        

