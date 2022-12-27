import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import glob
import matplotlib
import random as rand
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid
import csv
import image_processing as ip
import matplotlib as mpl



def boxplots_cloud_fraction(csv_path):
    df = pd.read_csv(csv_path)
    df.rename(columns={"max length scale": "maxls", "mean length scale": "meanls"},inplace=True)
    df['fraction'] = df['fraction'].fillna(0)
    cluster_indices = []
    for cluster in range(7):
        tmp = np.where(df.cluster == cluster + 1)
        cluster_indices.append(tmp[0])
    cloud_fractions = []
    for cluster in cluster_indices:
        tmp = df.fraction[cluster]
        cloud_fractions.append(tmp)
    fig = plt.figure(figsize =(10,7))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(cloud_fractions, patch_artist = True, notch ='True', showfliers=False)
 
    # changing color and linewidth of caps
    for cap in bp['caps']:
        cap.set(color ='#8B008B', linewidth = 2)
    
    # changing color and linewidth of medians
    for median in bp['medians']:
        median.set(color ='red', linewidth = 3)
    
    # changing style of fliers
    for flier in bp['fliers']:
        flier.set(marker ='D', color ='#e7298a', alpha = 0.5)
        
    # x-axis labels
    ax.set_xticklabels(['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Cluster 6', 'Cluster 7'])
    
    # Adding title
    plt.title("Cloud Fractions per Cluster", fontweight='bold')
    
    # Removing top axes and right axes ticks
    # ax.get_xaxis().tick_bottom()
    # ax.get_yaxis().tick_left()
    # x_grid_points = np.linspace(0,1,num=21)
    # ax.xaxis.set_ticks(x_grid_points)
    # plt.grid()
    plt.ylabel('Cloud Fraction', fontweight='bold')
    plt.tight_layout()
    plt.savefig('boxplots_cloud_fraction.eps')
    plt.savefig('boxplots_cloud_fraction.png')
    plt.show()



def boxplots_meanls(csv_path):
    df = pd.read_csv(csv_path)
    df.rename(columns={"max length scale": "maxls", "mean length scale": "meanls"},inplace=True)
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

    # changing color and linewidth of caps
    for cap in bp['caps']:
        cap.set(color ='#8B008B', linewidth = 2)
    
    # changing color and linewidth of medians
    for median in bp['medians']:
        median.set(color ='red', linewidth = 3)
    
    # changing style of fliers
    for flier in bp['fliers']:
        flier.set(marker ='D', color ='#e7298a', alpha = 0.5)
        
    # x-axis labels
    ax.set_yticklabels(['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Cluster 6', 'Cluster 7'])
    
    # Adding title
    plt.title("Mean Length Scale per Cluster")
    
    # Removing top axes and right axes ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    x_grid_points = np.linspace(0,128,num=17)
    ax.xaxis.set_ticks(x_grid_points)
    plt.grid()
    plt.xlabel('Mean Length Scale')
    plt.tight_layout()
    plt.savefig('boxplots_meanls.eps')
    plt.show()



def boxplots_maxls(csv_path):
    df = pd.read_csv(csv_path)
    df.rename(columns={"max length scale": "maxls", "mean length scale": "meanls"},inplace=True)
    df['maxls'] = df['maxls'].fillna(0)
    cluster_indices = []
    for cluster in range(7):
        tmp = np.where(df.cluster == cluster + 1)
        cluster_indices.append(tmp[0])
    maxls = []
    for cluster in cluster_indices:
        tmp = df.maxls[cluster]
        maxls.append(tmp)
    fig = plt.figure(figsize =(10, 7))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(maxls, patch_artist = True, notch ='True', vert = 0, showfliers=False)

    # changing color and linewidth of caps
    for cap in bp['caps']:
        cap.set(color ='#8B008B', linewidth = 2)
    
    # changing color and linewidth of medians
    for median in bp['medians']:
        median.set(color ='red', linewidth = 3)
    
    # changing style of fliers
    for flier in bp['fliers']:
        flier.set(marker ='D', color ='#e7298a', alpha = 0.5)
        
    # x-axis labels
    ax.set_yticklabels(['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Cluster 6', 'Cluster 7'])
    
    # Adding title
    plt.title("Max Length Scale per Cluster")
    
    # Removing top axes and right axes ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    x_grid_points = np.linspace(0,128,num=17)
    ax.xaxis.set_ticks(x_grid_points)
    plt.grid()
    plt.xlabel('Max Length Scale')
    plt.tight_layout()
    plt.savefig('boxplots_maxls.eps')
    plt.show()


def boxplots_maxls_meanls(csv_path):
    df = pd.read_csv(csv_path)
    df.rename(columns={"max length scale": "maxls", "mean length scale": "meanls"},inplace=True)
    df['maxls'] = df['maxls'].fillna(0)
    df['meanls'] = df['meanls'].fillna(0)
    cluster_indices = []
    for cluster in range(7):
        tmp = np.where(df.cluster == cluster + 1)
        cluster_indices.append(tmp[0])
    max_mean_ls = []
    for cluster in cluster_indices:
        tmp1 = df.maxls[cluster]
        max_mean_ls.append(tmp1)
        tmp2 = df.meanls[cluster]
        max_mean_ls.append(tmp2)
    fig = plt.figure(figsize =(20, 12))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(max_mean_ls, patch_artist = True, notch ='True', showfliers=False)

    # changing color and linewidth of caps
    for cap in bp['caps']:
        cap.set(color ='#8B008B', linewidth = 2)
    
    # changing color and linewidth of medians
    for median in bp['medians']:
        median.set(color ='red', linewidth = 3)
    
    # changing style of fliers
    for flier in bp['fliers']:
        flier.set(marker ='D', color ='#e7298a', alpha = 0.5)
        
    # x-axis labels
    ax.set_xticklabels(['Cluster 1\nMax', 'Cluster 1\nMean', 'Cluster 2\nMax', 'Cluster 2\nMean', 'Cluster 3\nMax', 'Cluster 3\nMean', 'Cluster 4\nMax', 'Cluster 4\nMean', 'Cluster 5\nMax', 'Cluster 5\nMean', 'Cluster 6\nMax', 'Cluster 6\nMean', 'Cluster 7\nMax', 'Cluster 7\nMean'], fontsize=18)
    
    # y-axis labels
    # ax.set_yticklabels([0, 16, 32, 48, 64, 80, 96, 112, 128], fontsize=18)
    plt.yticks(fontsize=18)

    # Adding title
    plt.title("Max & Mean Length Scale per Cluster", fontweight='bold', fontsize=24)
    
    # Removing top axes and right axes ticks
    # ax.get_xaxis().tick_bottom()
    # ax.get_yaxis().tick_left()
    # x_grid_points = np.linspace(0,128,num=17)
    # ax.xaxis.set_ticks(x_grid_points)
    # plt.grid()
    plt.ylabel('Max & Mean Length Scale', fontweight='bold', fontsize=22)
    plt.tight_layout()
    plt.savefig('boxplots_maxls_meanls.eps')
    plt.savefig('boxplots_maxls_meanls.png')
    plt.show()



def boxplots_iorg(csv_path):
    df = pd.read_csv(csv_path)
    df.rename(columns={"max length scale": "maxls", "mean length scale": "meanls"},inplace=True)
    df['iOrg'] = df['iOrg'].fillna(0)
    cluster_indices = []
    for cluster in range(7):
        tmp = np.where(df.cluster == cluster + 1)
        cluster_indices.append(tmp[0])
    iOrgs = []
    for cluster in cluster_indices:
        tmp = df.iOrg[cluster]
        iOrgs.append(tmp)
    fig = plt.figure(figsize =(10,7))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(iOrgs, patch_artist = True, notch ='False', showfliers=False)
 
    # changing color and linewidth of caps
    for cap in bp['caps']:
        cap.set(color ='#8B008B', linewidth = 2)
    
    # changing color and linewidth of medians
    for median in bp['medians']:
        median.set(color ='red', linewidth = 3)
    
    # changing style of fliers
    for flier in bp['fliers']:
        flier.set(marker ='D', color ='#e7298a', alpha = 0.5)
        
    # x-axis labels
    ax.set_xticklabels(['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Cluster 6', 'Cluster 7'])
    
    # Adding title
    plt.title("iOrg per Cluster", fontweight='bold')
    
    # Removing top axes and right axes ticks
    # ax.get_xaxis().tick_bottom()
    # ax.get_yaxis().tick_left()
    # x_grid_points = np.linspace(0,1,num=21)
    # ax.xaxis.set_ticks(x_grid_points)
    # plt.grid()
    plt.ylabel('iOrg', fontweight='bold')
    plt.tight_layout()
    plt.savefig('boxplots_iOrg.eps')
    plt.savefig('boxplots_iOrg.png')
    plt.show()


def boxplots_cod(csv_path):
    df = pd.read_csv(csv_path)
    # df.rename(columns={"max length scale": "maxls", "mean length scale": "meanls"},inplace=True)
    df['cot_mean'] = df['cot_mean'].fillna(0)
    cluster_indices = []
    for cluster in range(7):
        tmp = np.where(df.cluster == cluster + 1)
        cluster_indices.append(tmp[0])
    cod = []
    for cluster in cluster_indices:
        tmp = df.cot_mean[cluster]
        cod.append(tmp)
    fig = plt.figure(figsize =(10, 7))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(cod, patch_artist = True, notch ='True', showfliers=False)
 
    # changing color and linewidth of caps
    for cap in bp['caps']:
        cap.set(color ='#8B008B', linewidth = 2)
    
    # changing color and linewidth of medians
    for median in bp['medians']:
        median.set(color ='red', linewidth = 3)
    
    # changing style of fliers
    for flier in bp['fliers']:
        flier.set(marker ='D', color ='#e7298a', alpha = 0.5)
        
    # x-axis labels
    ax.set_xticklabels(['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Cluster 6', 'Cluster 7'])
    
    # Adding title
    plt.title("Image Mean Cloud Optical Depth per Cluster", fontweight='bold')
    
    # Removing top axes and right axes ticks
    # ax.get_xaxis().tick_bottom()
    # ax.get_yaxis().tick_left()
    # x_grid_points = np.linspace(0,1,num=21)
    # ax.xaxis.set_ticks(x_grid_points)
    # plt.grid()
    plt.ylabel('Mean Cloud Optical Depth', fontweight='bold')
    plt.tight_layout()
    plt.savefig('boxplots_cod.eps')
    plt.savefig('boxplots_cod.png')
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


#------Plot Clustering------------------------------------------------------------------------------------------------------
def plot_clustering_csv(path, cmap, image_type, rows, columns):
    print('----------------------------------------------------------------------------------------------------')
    print('Creating plot.')
    df  = pd.read_csv(path)
    fig = plt.figure(figsize=(columns * 128, rows * 128)) 
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
                ax.imshow(Image.open(filenames[sample_list[sample_counter]]),cmap=cmap)
            elif image_type == 'cm':  
                ax.imshow(np.load(filenames[sample_list[sample_counter]]),cmap=cmap, vmin=0, vmax=1)
            elif image_type == 'cot':
                ax.imshow(np.load(filenames[sample_list[sample_counter]]), cmap=cmap, vmin=0, vmax=150)                    
            column_counter += 1
        row_counter += 1
        column_counter = 0
    plt.subplots_adjust(0,0,1,1,0.05,0.1)
    plt.savefig("images_per_cluster_2.png", dpi=10)
    print('Plot saved.')
    print('----------------------------------------------------------------------------------------------------')
    print()


#------Plot Clustering normalized images------------------------------------------------------------------------------------
def plot_clustering_csv_norm(path, cmap, image_type):
    print('----------------------------------------------------------------------------------------------------')
    print('Creating plot.')
    df  = pd.read_csv(path)
    rows = 7                                                                                        
    columns = 20
    fig = plt.figure(figsize=(columns * 32, rows * 32))                                           
    row_counter = 0
    column_counter = 0
    filenames = df.path
    images = []
    if image_type == 'cod':
        for i in filenames: 
            img_tmp = Image.open(i)
            images.append(np.log(np.array(img_tmp) + 1)) 
            img_tmp.close() 
    else:
        for i in filenames:#
            images.append(np.log(np.load(i) + 1))  
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
    plt.savefig("TEST_CSV2.png", dpi=1)
    print('Plot saved.')
    print('----------------------------------------------------------------------------------------------------')
    print()



def plot_cluster_range_csv(path, cmap, image_type):
    rows = 7
    columns = 5
    fig = plt.figure(figsize=(rows * 128, columns * 128))
    grid = ImageGrid(fig, 111, nrows_ncols=(rows, columns), axes_pad=1)
    df = pd.read_csv(path)
    filenames = df.path
    images = []
    for cluster in range(rows):
        tmp1 = np.where(df.center == cluster + 1)
        tmp1_list = tmp1[0].tolist()
        images.append(np.load(filenames[tmp1_list[0]]))
        tmp2 = np.where(df.cluster == cluster + 1)
        tmp2_list = tmp2[0].tolist()
        sample_list = rand.sample(tmp2_list, min(columns - 1, len(tmp2_list)))
        for sample in sample_list:
            images.append(np.load(filenames[sample]))
    for ax, im in zip(grid, images):
        ax.imshow(im, cmap=cmap, vmin=0, vmax=1)
        ax.set_ylabel('Test', fontsize=12)
    plt.savefig("TEST_NEW.png", dpi=10)



def plot_cluster_range_csv2(path, cmap, rows, columns, norm, plot_norm):
    fig, axes = plt.subplots(rows, columns,sharex=True, sharey=True)
    fig.set_figheight(rows*256/100)
    fig.set_figwidth(columns*256/100)
    plt.subplots_adjust(0,0,1,0.95,0.01,0.1)
    df = pd.read_csv(path)
    filenames = df.path
    images = []
    tmp_glob_max = []
    tmp_glob_min = []
    for file in filenames:
        tmp_glob_max.append(np.amax(np.load(file)))
        tmp_glob_min.append(np.amin(np.load(file)))
    glob_max = max(tmp_glob_max)
    glob_min = min(tmp_glob_min)
    for cluster in range(rows):
        tmp1 = np.where(df.center == cluster + 1)
        tmp1_list = tmp1[0].tolist()
        if norm:
            images.append(np.divide(np.log(np.load(filenames[tmp1_list[0]]) + 1), np.log(glob_max + 1)))
        else:
            images.append(np.load(filenames[tmp1_list[0]]))
        tmp2 = np.where(df.cluster == cluster + 1)
        tmp2_list = tmp2[0].tolist()
        sample_list = rand.sample(tmp2_list, min(columns - 1, len(tmp2_list)))
        for sample in sample_list:
            if norm:
                images.append(np.divide(np.log(np.load(filenames[sample]) + 1), np.log(glob_max + 1)))
            else:
                images.append(np.load(filenames[sample]))
    center_counter = 0
    counter = 0
    cluster = 1
    for ax, im in zip(axes.flat, images):
        if plot_norm:
            ax.imshow(im, cmap=cmap, vmin=0, vmax=1)
        else:
            ax.imshow(im, cmap=cmap, vmin=0, vmax=150)
        ax.set_xticks([])
        ax.set_yticks([])
        if counter == 0:
            ax.set_title('Center', fontsize=24, fontweight='bold')
            counter += 1
        if center_counter == 0:
            ax.set_ylabel('Cluster ' + str(cluster), fontsize=24, fontweight='bold')
            center_counter += 1
            cluster += 1
        elif center_counter == columns - 1:
            center_counter = 0
        else:
            center_counter += 1
    plt.tight_layout()
    if norm:
        plt.savefig("images_per_cluster_norm.png", dpi=100)
        plt.savefig("images_per_cluster_norm.eps", dpi=100)
    else:
        plt.savefig("images_per_cluster.png", dpi=100)
        plt.savefig("images_per_cluster.eps", dpi=100)




def plot_images_treshs(images, images_norm):
    print('----------------------------------------------------------------------------------------------------')
    print('Creating plot.')
    rows = len(images)                                                                                        
    treshs = [0.4, 0.3, 0.2, 0.1]
    columns = len(treshs) + 2
    images_treshs = []
    for image, image_norm in zip(images, images_norm):
        images_treshs.append(image)
        images_treshs.append(image_norm)
        for tresh in treshs:
            images_treshs.append(np.where(image_norm > tresh, 1, 0))
    fig = plt.figure(figsize=(rows * 16, columns * 16))
    grid = ImageGrid(fig, 111, nrows_ncols=(rows, columns), axes_pad=0.1)
    counter = 1
    for img, ax in zip(images_treshs, grid):
        if counter == 1:
            ax.imshow(img, cmap='Spectral', vmin=0, vmax=150)
            counter += 1
        else:
            ax.imshow(img, cmap='Spectral', vmin=0, vmax=1)
            counter += 1
        if counter == 7:
            counter = 1
        plt.axis('off')
    plt.savefig("TEST_TRESHS.png")
    print('Plot saved.')
    print(np.sum(images_treshs[2]))
    print(np.sum(images_treshs[3]))
    print(np.sum(images_treshs[4]))
    print(np.sum(images_treshs[5]))
    print('----------------------------------------------------------------------------------------------------')
    print()

    

def merge_physicals(path1, path2):
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    df3 = pd.merge(df1,df2[['sds_mean','path']])
    df4 = pd.merge(df3,df2[['trs_mean','path']])
    df5 = pd.merge(df4,df2[['cot_mean','path']])
    df6 = pd.merge(df5,df2[['ctp_mean','path']])
    df6.to_csv('TEST1.csv', index=False)


def plot_cod_distr(path):
    df  = pd.read_csv(path)
    filenames = df.path
    glob_max = 0
    glob_min = 255
    for file in filenames:
        tmp_arr = np.load(file)
        tmp_min = np.amin(tmp_arr).astype(int)
        tmp_max = np.amax(tmp_arr).astype(int)
        if tmp_min < glob_min:
            glob_min = tmp_min
        if tmp_max > glob_max:
            glob_max = tmp_max
    print('glob max: ' + str(glob_max))
    print('glob min: ' + str(glob_min))
    sum_hist = np.zeros(glob_max + 1) # --> + 1 needed if glob_min=0 (values 0 to glob_max)
    for file in filenames:
        hist, bins = np.histogram(np.load(file), bins=np.arange(glob_max + 2))
        sum_hist += hist
    plt.bar(bins[:glob_max + 1], sum_hist, 1.0)
    plt.ylabel('Frequency', fontweight='bold')
    plt.xlabel('COD Values', fontweight='bold')
    plt.title('Distribution of COD Values', fontweight='bold')
    plt.savefig('COD_distr.png')
    plt.savefig('COD_distr.eps')


def plot_cmap(cmap):
    fig, ax = plt.subplots(figsize=(10,1))
    fig.subplots_adjust(bottom=0.5)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation='horizontal')
    ax.set_xlabel('Cloud Optical Depth', fontweight='bold')
    plt.savefig('cmap.eps')
    plt.savefig('cmap.png')