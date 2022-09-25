from re import I
from numba import njit
import numpy as np
import matplotlib.pyplot as plt
import random as rand
import ot
import time

#------Faster function for calculation relabeling costs of all orientations-------------------------------------------------
@njit(fastmath=True)
def diff(arr1,arr2):
    dim = len(arr1)
    d=np.zeros(8)
    for i in range(dim):
        for j in range(len(arr1[0])):
            d[0] += np.abs(arr1[i][j]-arr2[i][j])               # default
            d[1] += np.abs(arr1[dim-j-1][i]-arr2[i][j])         # rot 90 left
            d[2] += np.abs(arr1[dim-i-1][dim-j-1]-arr2[i][j])   # rot 180 left
            d[3] += np.abs(arr1[j][dim-i-1]-arr2[i][j])         # rot 270 left
            d[4] += np.abs(arr1[i][dim-j-1]-arr2[i][j])         # default flipped
            d[5] += np.abs(arr1[dim-j-1][dim-i-1]-arr2[i][j])   # rot 90 left flipped
            d[6] += np.abs(arr1[dim-i-1][j]-arr2[i][j])         # rot 180 left flipped
            d[7] += np.abs(arr1[j][i]-arr2[i][j])               # rot 270 left flipped
    return np.min(d)


#------Function to divide images into patches---------------------------------------------------------------------------
def get_patches(image, patchsize, step):
    patches = []        # Empty list for storing all patches of the input image  
    for i in range(0,len(image)-patchsize+1,step):
        for j in range(len(image[0])-patchsize+1):
            patches.append(image[i:i+patchsize,j:j+patchsize])     # Appending all possible patches        
    return patches


#------Function for approximating GED---------------------------------------------------------------------------------------
def app_ged(arr1,arr2,bool):
    ged = diff(arr1,arr2)
    return ged


#------Patch Clustering of all images with algorithm of Gonzalez------------------------------------------------------------
def gonzalez_patches(patches, k, bool):
    
    heads = []                  # List for cluster heads
    heads_DM = np.zeros((k,k)).astype(int)  # Distance matrix for distances between heads
    cluster = []                # List for cluster IDs
    distributions = []          # List for the distributions of all images --> How many patches of the image belong to which cluster
    dist_to_head = []           # List to store the distance of each patch to its cluster's head
    initial = True              # If initial all patch to head distances have to be calculated
    
    heads.append(patches[0][0])     # Initial head is the very first patch of the first image --> Maybe there is a better choice

    
    for i in range(len(patches)):
        cluster.append(np.zeros(len(patches[i])).astype(int))   # For every image i append zero array of length: # patches in one image
    
    for i in range(len(patches)):
        dist_to_head.append([])     # For every image i append list for distances of patches to its cluster's head
                
    for i in range(len(patches)):       # Create distributions for each image
        distr = np.zeros(k).astype(int)
        distr[0] = len(patches[i])      # Starting with all the mass at the first cluster
        distributions.append(distr)
        
    print('----------------------------------------------------------------------------------------------------')
    print('Patch Clustering running:')
        
    for i in range(k-1):
        time_start = time.time()
        print('Iteration: ' + str(i + 1) + ' of ' + str(k-1))
        max_ic_dist = 0
        image_index = 0
        patch_index = 0
        if initial:
            for j in range(len(patches)):
                for l in range(len(patches[0])):
                    dist_to_head[j].append(app_ged(patches[j][l],heads[0],bool))
                    initial = False
            for j in range(len(patches)):
                if max(dist_to_head[j]) > max_ic_dist:
                    max_ic_dist = max(dist_to_head[j])
                    image_index = j
                    patch_index = dist_to_head[j].index(max_ic_dist)
        else:
            for j in range(len(patches)):
                if max(dist_to_head[j]) > max_ic_dist:
                    max_ic_dist = max(dist_to_head[j])
                    image_index = j
                    patch_index = dist_to_head[j].index(max_ic_dist)
        heads.append(patches[image_index][patch_index])

        for l in range(len(patches)):
            for m in range(len(patches[0])):
                temp1 = dist_to_head[l][m]
                temp2 = app_ged(patches[l][m],heads[i + 1],bool)
                if temp1 >= temp2:
                    distributions[l][int(cluster[l][m])] -= 1
                    distributions[l][i+1] += 1
                    cluster[l][m] = i+1
                    dist_to_head[l][m] = temp2
        time_end = time.time()
        print('Laufzeit:' + str(time_end - time_start))
                    
    print('----------------------------------------------------------------------------------------------------')
    print('Heads Distance Matrix calculated:')
    
    for i in range(len(heads)):
        for j in range(i+1,len(heads)):
            heads_DM[i][j] = app_ged(heads[i],heads[j],bool)
            
    heads_DM = heads_DM + np.transpose(heads_DM)
    
    print('----------------------------------------------------------------------------------------------------')
            
    return heads,heads_DM,distributions,cluster


#------Plot Clustering------------------------------------------------------------------------------------------------------
def plot_clustering(patches, cluster, k_start, k_end, split, name):
    
    k = k_end - k_start
    cluster_counter = np.zeros(k)
    cluster_size = 30
    fig = plt.figure(figsize=(cluster_size * 8, k * 8))
    
    for i in range(k_start, k_end):
        print('Plot Cluster: ' + str(i))
        indices = np.where(cluster == i)
        temp = np.count_nonzero(cluster == i)
        if temp < cluster_size:
            for j in indices[0]:
                if k_start == 0:
                    fig.add_subplot(k, cluster_size, i * cluster_size + int(cluster_counter[i] + 1))
                    plt.imshow(patches[j])
                    plt.axis('off')
                    cluster_counter[i] += 1
                else:
                    fig.add_subplot(k, cluster_size, (i - k) * cluster_size + int(cluster_counter[i - k] + 1))
                    plt.imshow(patches[j])
                    plt.axis('off')
                    cluster_counter[i - k] += 1
        else:
            for j in range(cluster_size):
                if k_start == 0:
                    l = rand.choice(indices[0])
                    fig.add_subplot(k, cluster_size, i * cluster_size + int(cluster_counter[i] + 1))
                    plt.imshow(patches[l])
                    plt.axis('off')
                    cluster_counter[i] += 1
                else:
                    l = rand.choice(indices[0])
                    fig.add_subplot(k, cluster_size, (i - k) * cluster_size + int(cluster_counter[i - k] + 1))
                    plt.imshow(patches[l])
                    plt.axis('off')
                    cluster_counter[i - k] += 1
                    
    plt.subplots_adjust(0,0,1,1,0.1,0.1)
    plt.savefig(name+".png")
    #plt.show()
    
    
#------Plot Clustering------------------------------------------------------------------------------------------------------
def plot_clustering2(patches, cluster, k_start, k_end, name):
    k = k_end - k_start
    cluster_size = 30
    indices = []
    fig = plt.figure(figsize=(cluster_size * 8, k * 8))
    row_counter = 0
    column_counter = 0
    for i in range(k_start, k_end):
        indices.append(np.where(cluster == i))
    for i in indices:
        l = rand.choice(i, cluster_size)
        for j in range(len(l)):
            fig.add_subplot(k, cluster_size, row_counter * cluster_size + column_counter)
            plt.imshow(patches[l[j]])
            column_counter += 1
        row_counter += 1
    plt.subplots_adjust(0,0,1,1,0.1,0.1)
    plt.savefig(name+".png")
    

#------Function to compute Wasserstein Distance-----------------------------------------------------------------------------
def wasserstein_dist(distr1, distr2, CM):
    wd = 0
    if len(distr1) == len(distr2) and len(distr1) == len(CM):
        wd = ot.emd2(distr1, distr2, CM)
    else:
        print('Error WD computation!')    
    return wd


#------Image Clustering based on Gonzalez's Algorithm-----------------------------------------------------------------------
def gonzalez_images(k, distributions, CM):
    
    heads = [] 
    heads.append(distributions[0])
    dist_to_head = []
    cluster = np.zeros(len(distributions)).astype(int)
    initial = True
    
    print('----------------------------------------------------------------------------------------------------')
    print('Image Clustering running:')
    
    for i in range(k - 1):
        print('Iteration: ' + str(i + 1))
        if initial:
            for j in range(len(distributions)):
                dist_to_head.append(wasserstein_dist(distributions[j], heads[int(cluster[j])], CM))
                max_ic_dist = max(dist_to_head)
                index = dist_to_head.index(max_ic_dist)
                initial = False
        else:
            max_ic_dist = max(dist_to_head)
            index = dist_to_head.index(max_ic_dist)
            initial = False
        cluster[index] = i + 1
        heads.append(distributions[index])
        for l in range(len(distributions)):
            temp1 = dist_to_head[l]
            temp2 = wasserstein_dist(distributions[l], heads[i + 1], CM)
            if temp1 >= temp2:
                cluster[l] = i + 1
                dist_to_head[l] = temp2
                
    print('----------------------------------------------------------------------------------------------------')
    
    return cluster


#------Image Clustering based on a k-median Clustering----------------------------------------------------------------------
def d2_images(k1, distributions, CM, stop):
    
    cluster = np.zeros(len(distributions)).astype(int)
    center = np.zeros(k1).astype(int)
    ic_dist = np.zeros(k1)
    start = np.random.randint(0, len(distributions), k1)
    #start = [0, 1, 2]
    
    for i in range(len(center)):
        center[i] = start[i]

    print('----------------------------------------------------------------------------------------------------')
    print('Image Clustering running:')

    for i in range(stop):
        print('Iteration: ' + str(i + 1) + ' of ' + str(stop))
        print('Assignment to cluster.')
        for j in range(len(distributions)):
            initial = True
            temp1 = 0
            for k in range(len(center)):
                if initial:
                    temp1 = wasserstein_dist(distributions[j], distributions[center[k]], CM)
                    cluster[j] = k
                    initial = False
                else:
                    temp2 = wasserstein_dist(distributions[j], distributions[center[k]], CM)
                    if temp2 < temp1:
                        cluster[j] = k
                        temp1 = temp2
        for j in range(k1):
            print('Iteration ' + str(i) + ':\tCalculation of Centroid\t' + str(j + 1) + '.')
            initial = True
            indices = np.where(cluster == j)
            ic_dist = 0
            index = 0
            for k in indices[0]:
                temp = 0
                for l in indices[0]:
                    temp += wasserstein_dist(distributions[k], distributions[l], CM)
                if initial:
                    ic_dist = temp
                    index = k
                    initial = False
                elif temp < ic_dist:
                    ic_dist = temp
                    index = k
        center[j] = index
    
    print('----------------------------------------------------------------------------------------------------')

    return cluster, center