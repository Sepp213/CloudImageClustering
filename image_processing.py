from numba import njit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random as rand
import ot
import time
import glob
import random
import cloudmetrics
import csv
from PIL import Image
from torchvision import transforms

#------Faster function for calculation relabeling costs of all orientations-------------------------------------------------
@njit(fastmath=True)
def app_ged(arr1,arr2):
    assert (len(arr1) == len(arr2)), "Dimensions of patches are not equal!"
    assert (len(arr1[0]) == len(arr2[0])), "Dimensions of patches are not equal!"
    dimension = len(arr1)
    distance = np.zeros(8)
    for row in range(dimension):
        for column in range(len(arr1[0])):
            distance[0] += np.abs(arr1[row][column]-arr2[row][column])                            # default
            distance[1] += np.abs(arr1[dimension-column-1][row]-arr2[row][column])                # rotate 90 left
            distance[2] += np.abs(arr1[dimension-row-1][dimension-column-1]-arr2[row][column])    # rotate 180 left
            distance[3] += np.abs(arr1[column][dimension-row-1]-arr2[row][column])                # rotate 270 left
            distance[4] += np.abs(arr1[row][dimension-column-1]-arr2[row][column])                # default flipped
            distance[5] += np.abs(arr1[dimension-column-1][dimension-row-1]-arr2[row][column])    # rotate 90 left and flipped
            distance[6] += np.abs(arr1[dimension-row-1][column]-arr2[row][column])                # rotate 180 left and flipped
            distance[7] += np.abs(arr1[column][row]-arr2[row][column])                            # rotate 270 left and flipped
    return np.min(distance)


#------Function to divide images into patches---------------------------------------------------------------------------
def get_patches(image, patchsize, step_size):
    assert (patchsize >= step_size), "The step size is larger than the patches, so the whole image is not covered by patches!"
    patches = []                                                                # Empty list for storing all patches of the input image  
    for row in range(0,len(image)-patchsize+1,step_size):                       # One patch for each pixel of a row
        for column in range(len(image[0])-patchsize+1):                         # One patch for each pixel of a column
            patches.append(image[row:row+patchsize,column:column+patchsize])    # Appending all possible patches        
    return patches


#------Patch Clustering to minimize intercluster distance with algorithm of Gonzalez----------------------------------------
def gonzalez_patches(patches, k):                                               # patches --> List of list of arrays (in outer list is one list of arrays for each image), k --> # of patch cluster
    heads = []                                                                  # List for cluster heads
    cluster = []                                                                # List for cluster IDs
    distributions = []                                                          # List for the distributions of all images --> How many patches of the image belong to which cluster
    dist_to_head = []                                                           # List to store the distance of each patch to its cluster's head
    initial = True                                                              # If initial all patch to head distances have to be calculated
    heads.append(patches[0][0])                                                 # Initial head is the very first patch of the first image
    for image in range(len(patches)):
        cluster.append(np.zeros(len(patches[image])).astype(int))               # For each image append zero array of length # patches in one image --> For each patch the image and cluster information is stored this way
    for image in range(len(patches)):
        dist_to_head.append([])                                                 # For each image append empty list for distances of patches to its cluster's head
    for image in range(len(patches)):                                           # Create distributions for each image
        temp = np.zeros(k)                                                      # Each distribution contains k entries
        temp[0] = len(patches[image])                                           # Starting with all the mass at the first cluster
        distributions.append(temp)                                              # Appending all distributions to one list
    print('----------------------------------------------------------------------------------------------------')
    print('Patch Clustering running:')
    for iteration in range(k-1):
        time_start = time.time()
        max_ic_dist = 0
        image_index = 0
        patch_index = 0
        if initial:
            for image_counter in range(len(patches)):
                for patch_counter in range(len(patches[0])):
                    dist_to_head[image_counter].append(app_ged(patches[image_counter][patch_counter],heads[0]))             # Initially compute compute the distances of eachs patch to it's head and store the distance in dist_to_head
                    initial = False
            for image_counter in range(len(patches)):
                if max(dist_to_head[image_counter]) > max_ic_dist:                                                          # Find a image with larger intercluster distance
                    max_ic_dist = max(dist_to_head[image_counter])                                                          # Store the max intercluster distance                                                               
                    image_index = image_counter                                                                             # Store the image index
                    patch_index = dist_to_head[image_counter].index(max_ic_dist)                                            # Store the patch index within the patchlist of the image
        else:
            for image_counter in range(len(patches)):                                                                       # From the second iteration on, the existing distances to heads can be used.
                if max(dist_to_head[image_counter]) > max_ic_dist:                                                          # Find a image with larger intercluster distance
                    max_ic_dist = max(dist_to_head[image_counter])                                                          # Store the max intercluster distance
                    image_index = image_counter                                                                             # Store the image index
                    patch_index = dist_to_head[image_counter].index(max_ic_dist)                                            # Store the patch index within the patchlist of the image
        heads.append(patches[image_index][patch_index])                                                                     # The patch which maximizes the intercluster distance opens a new cluster (becomes next head)
        for image_counter in range(len(patches)):
            for patch_counter in range(len(patches[0])):                                                                    # The patches of each image need to be assigned to the new cluster if the distance to the new heads better
                temp1 = dist_to_head[image_counter][patch_counter]
                temp2 = app_ged(patches[image_counter][patch_counter],heads[iteration + 1])                                 # Compute distance to the new head
                if temp1 >= temp2:
                    distributions[image_counter][int(cluster[image_counter][patch_counter])] -= 1                           # Update distribution of the affected image
                    distributions[image_counter][iteration+1] += 1                                                          # Update distribution of the affected image
                    cluster[image_counter][patch_counter] = iteration+1                                                     # Update cluster of the 'moved' patch
                    dist_to_head[image_counter][patch_counter] = temp2                                                      # Update intercluster distance of the 'moved' patch
        time_end = time.time()
        print('Iteration:\t' + str(iteration + 1) + '\tof\t' + str(k-1) + '\tin\t' + str(time_end - time_start) + '\ts.')
    print('----------------------------------------------------------------------------------------------------')
    print()
    print('----------------------------------------------------------------------------------------------------')
    print('Calculation of Heads Distance Matrix', end=' ')
    heads_distance_matrix = np.zeros((k,k)).astype(int)                                                                     # Distance matrix for distances between heads
    for row in range(len(heads)):
        for column in range(row+1,len(heads)):
            heads_distance_matrix[row][column] = app_ged(heads[row],heads[column])                                          # Compute upper triangular matrix of distances between heads
    heads_distance_matrix = heads_distance_matrix + np.transpose(heads_distance_matrix)                                     # Create symmetric distance matrix
    print('done.')
    print('----------------------------------------------------------------------------------------------------')
    print()
    return heads_distance_matrix,distributions,cluster
    
    
#------Plot Clustering------------------------------------------------------------------------------------------------------
def plot_clustering(images, cluster, sample_size, k_start, k_end, name, color1, color2):
    print('----------------------------------------------------------------------------------------------------')
    print('Creating plot.')
    custom_cmap = matplotlib.colors.ListedColormap([color1, color2])                                # Create custom cmap from color 1 and color2
    k = k_end - k_start                                                                             # Calculate number of rows
    indices = []
    fig = plt.figure(figsize=(sample_size * 16, k * 16))                                            # Create figure with sample_size x k entries with 16x16 images
    row_counter = 0
    column_counter = 0
    for cluster_counter in range(k_start, k_end):
        indices.append(np.where(cluster == cluster_counter))                                        # Collect all indices for a cluster
    for cluster_counter in indices:
        index_list = cluster_counter[0].tolist()                                                    # Iterate through index lists of clusters
        if len(index_list) < sample_size:                                                           # If cluster is smaller than sample_size take all indices
            sample_list = index_list
        else:
            sample_list = rand.sample(index_list, sample_size)                                      # Else take a random sample of # sample_size indices
        for sample_counter in range(len(sample_list)):
            fig.add_subplot(k, sample_size, row_counter * sample_size + column_counter + 1)         # Add samples of cluster to subplot
            # plt.imshow(images[sample_list[sample_counter]], cmap=custom_cmap, vmin=0, vmax=1)       # Adding vmin and vmax --> Completly cloudy images get color of vmax (without vmin and vmax they got the same color as zeros in mixed images)
            plt.imshow(images[sample_list[sample_counter]],cmap='hot')                              # Test for COD Clustering
            plt.axis('off')
            column_counter += 1
        row_counter += 1
        column_counter = 0
    plt.subplots_adjust(0,0,1,1,0.1,0.1)
    plt.savefig(name+".png")
    print('Plot saved.')
    print('----------------------------------------------------------------------------------------------------')
    print()
    

#------Function to compute Wasserstein Distance-----------------------------------------------------------------------------
def wasserstein_dist(distr1, distr2, CM):
    wd = 0
    assert (len(distr1) == len(distr2) == len(CM)), "Distributions for Wasserstein Distance computation have the wrong length!"
    try:
        wd = ot.emd2(distr1, distr2, CM)            # Use emd2 from pot package to compute Wasserstein distance with custom cost matrix CM which comes from patch clustering
    except:
        print('Error WD computation!')    
    return wd


#------Image Clustering based on a k-median Clustering----------------------------------------------------------------------
def d2_images(k1, distributions, CM, stop): 
    cluster = np.zeros(len(distributions)).astype(int)
    ic_dist = np.zeros(k1)
    CCM = np.zeros((k1,k1))
    check = False
    while check == False:
        check = True
        print('check')
        start = rand.sample(range(len(distributions)), k1)
        for i in range(len(start)):
            for j in range(i + 1, len(start)):
                temp = wasserstein_dist(distributions[start[i]],distributions[start[j]],CM)
                CCM[i][j] = temp
                if temp == 0:
                    check = False
        if check == True:
            center = start
    print('----------------------------------------------------------------------------------------------------')
    print('Image Clustering running:')
    for i in range(stop):
        print('Iteration\t' + str(i + 1) + '\tof\t' + str(stop) + ':', end='\t')
        print('Assignment to cluster', end=' ')
        for j in range(len(distributions)):
            initial = True
            temp1 = 0
            for k in range(len(center)):
                if initial:
                    temp1 = wasserstein_dist(distributions[j], distributions[center[k]], CM)
                    cluster[j] = k
                else:
                    temp2 = wasserstein_dist(distributions[j], distributions[center[k]], CM)
                    if temp2 < temp1:
                        cluster[j] = k
                        temp1 = temp2
                initial = False
        print('done.', end='\t&\t')
        print('Calculation of centroids', end=' ')
        for j in range(k1):
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
                elif temp < ic_dist:
                    ic_dist = temp
                    index = k
                initial = False
            center[j] = index
        print('done.')
    print('----------------------------------------------------------------------------------------------------')
    print()
    return cluster, center


# ------Cloudmetrics Computation--------------------------------------------------------------------------------------------
def compute_cloudmetrics(sample, images, cluster_images, name):
    print('----------------------------------------------------------------------------------------------------')
    print('Cloudmetrics Computation running.')
    fieldnames = ['path', 'cluster', 'fraction', 'open sky max', 'open sky mean', 'orientation', 'max length scale', 'mean length scale', 'iOrg', 'woi3']
    rows = []
    for i,j,l in zip(sample, images, cluster_images):
        labels = cloudmetrics.objects.label(mask=j)
        try:
            tmp1 = cloudmetrics.mask.cloud_fraction(j)
        except:
            tmp1 = 0.0
        try:
            tmp2, tmp3 = cloudmetrics.mask.open_sky(j) 
        except:
            tmp2 = 0.0 
            tmp3 = 0.0
        try: 
            tmp4 = cloudmetrics.mask.orientation(j)
        except: 
            tmp4 = 0.0
        try:
            tmp5 = cloudmetrics.objects.max_length_scale(labels) # --> delete
        except:
            tmp5 = 0.0
        try: 
            tmp6 = cloudmetrics.objects.mean_length_scale(labels)
        except:
            tmp6 = 0.0
        try:
            tmp7 = cloudmetrics.objects.metrics.iorg(labels)
        except:
            tmp7 = 0.0
        row = {'path': i, 'cluster': l, 'fraction': tmp1, 'open sky max': tmp2, 'open sky mean': tmp3, 'orientation': tmp4, 'max length scale': tmp5, 'mean length scale': tmp6, 'iOrg': tmp7}
        rows.append(row)
    print('Cloudmetrics successfully computed.')
    print('----------------------------------------------------------------------------------------------------')
    print()
    print('----------------------------------------------------------------------------------------------------')
    print('Writing results to csv.')
    with open('D2_metrics_' + name + '.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print('File saved.')
    print('----------------------------------------------------------------------------------------------------')
    print()



# ------CMASK Clustering----------------------------------------------------------------------------------------------------
def cmask_clustering(sample_size, k_images, d2_stop, k_patches, patchsize, stepsize):
    filenames = glob.glob("/Users/sz/Documents/Uni/Master/Masterarbeit/Masterarbeit_Projekt/ImageClusteringMain/data/cmask/germany_cmask_128x128_npy/*.npy")
    filenames.sort()
    sample = random.sample(filenames, sample_size)
    images = [np.load(arr).astype(int) for arr in sample]
    for i in patchsize:
        for j in k_patches:
            for m in stepsize[patchsize.index(i)]:
                patches = []
                for l in images:
                    patches.append(get_patches(l,i,m))
                name = 'k_images' + str(k_images) + '_k_patches' + str(j) + '_patchsize' + str(i) + '_stepsize' + str(m)
                heads_CM, distributions, cluster_patches = gonzalez_patches(patches, j)
                cluster_images, center_images = d2_images(k_images,distributions,heads_CM,d2_stop)
                plot_clustering(images, cluster_images, 30, 0, k_images, "D2_" + name, 'navy', 'whitesmoke')
                compute_cloudmetrics(sample, images, cluster_images, name)


# ------Cloud Optical Depth Clustering--------------------------------------------------------------------------------------
def cod_clustering(sample_size, k_images, d2_stop, k_patches, patchsize, stepsize):
    filenames = glob.glob("/Users/sz/Documents/Uni/Master/Masterarbeit/Masterarbeit_Projekt/ImageClusteringMain/data/cloud_optical_depth/gscale_128/random_crops/1/*.jpeg")
    filenames.sort()
    sample = random.sample(filenames, sample_size)
    # images = [image.imread(sample_image) for sample_image in sample]  
    images = [Image.open(sample_image) for sample_image in sample]
    # transform_norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.0112, 0.0770)])   # https://www.geeksforgeeks.org/how-to-normalize-images-in-pytorch/ --> mean and std calculated from DC for dataset
    # data = [np.array(transform_norm(image)) for image in images]
    # print(data[0])
    print(images[0])
    # for i in patchsize:
    #     for j in k_patches:
    #         for m in stepsize[patchsize.index(i)]:
    #             patches = []
    #             for l in data:
    #                 patches.append(get_patches(l,i,m))
    #             name = 'COD_k_images' + str(k_images) + '_k_patches' + str(j) + '_patchsize' + str(i) + '_stepsize' + str(m)
    #             heads_CM, distributions, cluster_patches = gonzalez_patches(patches, j)
    #             cluster_images, center_images = d2_images(k_images,distributions,heads_CM,d2_stop)
    #             plot_clustering(images, cluster_images, 30, 0, k_images, "D2_" + name, 'navy', 'whitesmoke')
    #             compute_cloudmetrics(sample, images, cluster_images, name)

