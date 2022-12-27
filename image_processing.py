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
from math import sqrt
import analysis as ana


#------Faster function for calculation relabeling costs of all orientations-------------------------------------------------
@njit
def Delta_final(arr1,arr2):
    assert (len(arr1) == len(arr2)), "Dimensions of patches are not equal!"
    assert (len(arr1[0]) == len(arr2[0])), "Dimensions of patches are not equal!"
    dimension = len(arr1)
    distance = np.zeros(8)
    for row in range(dimension):
        for column in range(len(arr1[0])):
            #default 
            distance[0] += np.abs(arr1[row][column]-arr2[row][column])  
            # default flipped
            distance[1] += np.abs(arr1[dimension-row-1][column]-arr2[row][column])
            # rot 90
            distance[2] += np.abs(arr1[column][dimension-row-1]-arr2[row][column]) 
            # rot 90 flipped
            distance[3] += np.abs(arr1[column][row]-arr2[row][column])
            # rot 180
            distance[4] += np.abs(arr1[dimension-row-1][dimension-column-1]-arr2[row][column])     
            # rot 180 flipped
            distance[5] += np.abs(arr1[row][dimension-column-1]-arr2[row][column])
            # rot 270                
            distance[6] += np.abs(arr1[dimension-column-1][row]-arr2[row][column])   
            # rot 270 flipped                                               
            distance[7] += np.abs(arr1[dimension-column-1][dimension-row-1]-arr2[row][column])                                               
    return np.min(distance)


# ------Function to calculate GED for splitted patches----------------------------------------------------------------------
@njit(fastmath=True)
def split_cost(arr1_l, arr1_r, arr2_l, arr2_r):
    distance = []
    # flip and rotate left patch half
    distance.append(split_distance(arr1_l, arr2_l) + split_distance(arr1_r, arr2_r))
    distance.append(split_rotate_distance(arr1_l, arr2_l) + split_distance(arr1_r, arr2_r))
    distance.append(split_fliph_distance(arr1_l, arr2_l) + split_distance(arr1_r, arr2_r))
    distance.append(split_fliph_rotate_distance(arr1_l, arr2_l) + split_distance(arr1_r, arr2_r))
    # flip and rotate left patch half
    distance.append(split_distance(arr1_r, arr2_r) + split_distance(arr1_l, arr2_l))
    distance.append(split_rotate_distance(arr1_r, arr2_r) + split_distance(arr1_l, arr2_l))
    distance.append(split_fliph_distance(arr1_r, arr2_r) + split_distance(arr1_l, arr2_l))
    distance.append(split_fliph_rotate_distance(arr1_r, arr2_r) + split_distance(arr1_l, arr2_l))
    return min(distance)


# ------Function to calculate distance of splitted patches------------------------------------------------------------------
@njit(fastmath=True)
def split_distance(arr1, arr2):
    distance = 0
    num_rows = len(arr1)
    num_col = len(arr1[0])
    assert (len(arr1) == len(arr2) and len(arr1[0]) == len(arr2[0])), 'Dimensions of splitted patches are not the same!'
    for row_counter in range(len(arr1)):
        for column_counter in range(len(arr1[0])):
            distance += np.abs(arr1[row_counter][column_counter]-arr2[row_counter][column_counter])
    return distance

# ------Function to calculate distance of a rotated, splitted patch---------------------------------------------------------
@njit(fastmath=True)
def split_rotate_distance(arr1, arr2):
    distance = 0
    num_rows = len(arr1)
    num_col = len(arr1[0])
    assert (len(arr1) == len(arr2) and len(arr1[0]) == len(arr2[0])), 'Dimensions of splitted patches are not the same!'
    for row_counter in range(len(arr1)):
        for column_counter in range(len(arr1[0])):
            distance += np.abs(arr1[num_rows - row_counter][num_col - column_counter]-arr2[row_counter][column_counter])
    return distance


# ------Function to calculate distance of a horizonatlly flipped, splitted patch--------------------------------------------
@njit(fastmath=True)
def split_fliph_distance(arr1, arr2):
    distance = 0
    num_rows = len(arr1)
    num_col = len(arr1[0])
    assert (len(arr1) == len(arr2) and len(arr1[0]) == len(arr2[0])), 'Dimensions of splitted patches are not the same!'
    for row_counter in range(len(arr1)):
        for column_counter in range(len(arr1[0])):
            distance += np.abs(arr1[row_counter][num_col - column_counter]-arr2[row_counter][column_counter])
    return distance


# ------Function to calculate distance of a horizonatlly flipped, rotatded, splitted patch----------------------------------
@njit(fastmath=True)
def split_fliph_rotate_distance(arr1, arr2):
    distance = 0
    num_rows = len(arr1)
    num_col = len(arr1[0])
    assert (len(arr1) == len(arr2) and len(arr1[0]) == len(arr2[0])), 'Dimensions of splitted patches are not the same!'
    for row_counter in range(len(arr1)):
        for column_counter in range(len(arr1[0])):
            distance += np.abs(arr1[num_rows - row_counter][column_counter]-arr2[row_counter][column_counter])
    return distance


#------Function to divide images into patches-------------------------------------------------------------------------------
def get_patches(image, patchsize, step_size):
    assert (patchsize >= step_size), "The step size is larger than the patches, so the whole image is not covered by patches!"
    # Empty list for storing all patches of the input image
    patches = []                                                                  
    for row in range(0,len(image)-patchsize+1,step_size):                       
        for column in range(len(image[0])-patchsize+1): 
            # Appending all possible patches                           
            patches.append(image[row:row+patchsize,column:column+patchsize])         
    return patches


#------Patch Clustering to minimize intercluster distance with algorithm of Gonzalez----------------------------------------
def gonzalez_patches(patches, k, eps):   
    # List for cluster heads                                        
    heads = []  
    # List for cluster IDs                                                                
    cluster = []    
    # List for the distributions of all images --> How many patches of the image belong to which cluster                                                            
    distributions = []   
    # List to store the distance of each patch to its cluster's head                                                       
    dist_to_head = []   
    # If initial all patch to head distances have to be calculated                                                        
    initial = True      
    # Initial head is the very first patch of the first image                                                        
    heads.append(patches[0][0])                                                 
    for image in range(len(patches)):
        # For each image append zero array of length # patches in one image --> For each patch the image and cluster information is stored this way
        cluster.append(np.zeros(len(patches[image])).astype(int))               
    for image in range(len(patches)):
        # For each image append empty list for distances of patches to its cluster's head
        dist_to_head.append([]) 
    # Create distributions for each image                                                
    for image in range(len(patches)):      
        # Each distribution contains k entries                                     
        temp = np.zeros(k)   
        # Starting with all the mass at the first cluster                                                   
        temp[0] = len(patches[image])    
        # Appending all distributions to one list                                       
        distributions.append(temp)                                              
    print('----------------------------------------------------------------------------------------------------')
    print('Patch Clustering running:')
    #for iteration in range(k-1):
    iteration = 0
    max_ic_dist = 64
    while iteration < k-1 and max_ic_dist > eps:
        time_start = time.time()
        max_ic_dist = 0
        image_index = 0
        patch_index = 0
        if initial:
            for image_counter in range(len(patches)):
                for patch_counter in range(len(patches[0])):
                    # Initially compute the distances of eachs patch to it's head and store the distance in dist_to_head
                    dist_to_head[image_counter].append(Delta_final(patches[image_counter][patch_counter],heads[0]))             
                    initial = False
            for image_counter in range(len(patches)):
                # Find a image with larger intercluster distance
                if max(dist_to_head[image_counter]) > max_ic_dist: 
                    # Store the max intercluster distance                                                         
                    max_ic_dist = max(dist_to_head[image_counter]) 
                    # Store the image index                                                                                                                        
                    image_index = image_counter 
                    # Store the patch index within the patchlist of the image                                                                            
                    patch_index = dist_to_head[image_counter].index(max_ic_dist)                                            
        else:
            # From the second iteration on, the existing distances to heads can be used.
            for image_counter in range(len(patches)):    
                # Find a image with larger intercluster distance                                                                   
                if max(dist_to_head[image_counter]) > max_ic_dist: 
                    # Store the max intercluster distance                                                         
                    max_ic_dist = max(dist_to_head[image_counter])       
                    # Store the image index                                                   
                    image_index = image_counter   
                    # Store the patch index within the patchlist of the image                                                                          
                    patch_index = dist_to_head[image_counter].index(max_ic_dist)                                            
        print('Maximum intercluster distance:\t' + str(max_ic_dist))
        # The patch which maximizes the intercluster distance opens a new cluster (becomes next head)
        heads.append(patches[image_index][patch_index])                                                                     
        counter_move = 0
        for image_counter in range(len(patches)):
            # The patches of each image need to be assigned to the new cluster if the distance to the new heads better
            for patch_counter in range(len(patches[0])):                                                                    
                temp1 = dist_to_head[image_counter][patch_counter]
                # Compute distance to the new head
                temp2 = Delta_final(patches[image_counter][patch_counter],heads[iteration + 1])                                 
                if temp1 >= temp2:
                    # Update distribution of the affected image
                    distributions[image_counter][int(cluster[image_counter][patch_counter])] -= 1   
                    # Update distribution of the affected image                        
                    distributions[image_counter][iteration+1] += 1     
                    # Update cluster of the 'moved' patch                                                     
                    cluster[image_counter][patch_counter] = iteration+1
                    # Update intercluster distance of the 'moved' patch                                                     
                    dist_to_head[image_counter][patch_counter] = temp2                                                      
                    counter_move += 1
        print('Patches moved:\t' + str(counter_move))
        time_end = time.time()
        iteration += 1
        print('Iteration:\t' + str(iteration + 1) + '\tof\t' + str(k-1) + '\tin\t' + str(time_end - time_start) + '\ts.')
    print('----------------------------------------------------------------------------------------------------')
    print()
    print('----------------------------------------------------------------------------------------------------')
    print('Calculation of Heads Distance Matrix', end=' ')
    distributions_shortened = []
    for distribution in distributions:
        distributions_shortened.append(distribution[range(len(heads))])
    # Distance matrix for distances between heads
    heads_distance_matrix = np.zeros((len(heads),len(heads))).astype(float)                                                                     
    for row in range(len(heads)):
        for column in range(row+1,len(heads)):
            # Compute upper triangular matrix of distances between heads
            heads_distance_matrix[row][column] = Delta_final(heads[row],heads[column])    
    # Create symmetric distance matrix                                              
    heads_distance_matrix = heads_distance_matrix + np.transpose(heads_distance_matrix)                                     
    print('done.')
    print('----------------------------------------------------------------------------------------------------')
    print()
    return heads_distance_matrix,distributions_shortened,cluster
    

#------Function to compute Wasserstein Distance-----------------------------------------------------------------------------
def wasserstein_dist(distr1, distr2, CM):
    wd = 0
    assert (len(distr1) == len(distr2) == len(CM)), "Distributions for Wasserstein Distance computation have the wrong length! Lengths (distr1/distr2/CM): " + str(len(distr1)) + '/' + str(len(distr2)) + '/' + str(len(CM))
    try:
        # Use emd2 from pot package to compute Wasserstein distance with custom cost matrix CM which comes from patch clustering
        wd = ot.emd2(distr1, distr2, CM)            
    except:
        print('Error WD computation!')    
    return wd


# ------Compute Wasserstein barycenter--------------------------------------------------------------------------------------
def ws_barycenter(distributions, CM):
    return ot.barycenter(distributions, CM, 1e-3, method='sinkhorn_stabilized') 


#------Image Clustering based on a D2 Clustering----------------------------------------------------------------------------
def d2_images(k1, distributions, CM, stop): 
    cluster = np.zeros(len(distributions)).astype(int)
    ic_dist = np.zeros(k1)
    CCM = np.zeros((k1,k1))
    check = False
    counter = 0
    while check == False and counter <= 1000:
        check = True
        print('check')
        start = rand.sample(range(len(distributions)), k1)
        for i in range(len(start)):
            for j in range(i + 1, len(start)):
                temp = wasserstein_dist(distributions[start[i]],distributions[start[j]],CM)
                CCM[i][j] = temp
                if temp == 0:
                    check = False
                    counter += 1
        if check == True:
            center = start
    assert (check == True), "Initial centers with distance > 0 could not be selected!"
    print('----------------------------------------------------------------------------------------------------')
    print('Image Clustering running:')
    bool_stop = True
    i = 0
    # for i in range(stop):
    while i < stop and bool_stop:
        time1 = time.time()
        bool_stop = False
        for j in range(len(distributions)):
            initial = True
            temp1 = 0
            cluster_start = cluster[j]
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
            if cluster_start != cluster[j]:
                bool_stop = True
        time2 = time.time()
        print('Assignment to centroids done:\t' + str(time2 - time1) + '\ts.')
        time3 = time.time()
        for j in range(k1):
            initial = True
            indices = np.where(cluster == j)
            ic_dist = 0
            index = 0
            counter = 1
            for k in indices[0]: 
                temp = 0
                for l in indices[0]:
                    temp += wasserstein_dist(distributions[k], distributions[l], CM)
                    print('Iteration\t' + str(i) + ':\tCalculation of centroid\t' + str(j) + '\t' + str(counter) + '\tof\t' + str(len(indices[0])**2))
                    counter += 1
                if initial:
                    ic_dist = temp
                    index = k
                elif temp < ic_dist:
                    ic_dist = temp
                    index = k
                initial = False
            center[j] = index
        time4 = time.time()
        print('Calculation of centroids done:\t' + str(time4 - time3) + '\ts.')
        i += 1
    print('----------------------------------------------------------------------------------------------------')
    print()
    return cluster, center


#------Image Clustering based on a D2 Clustering----------------------------------------------------------------------------
def d2_images_fast(k1, distributions, CM, stop):
    distributions_normalized = [] 
    for distr in distributions:
        # normalized distributions lead du worse results
        # distr_sum = np.sum(distr)
        # distributions_normalized.append(distr/distr_sum)  --> Better results for unnormalized distributions!
        distributions_normalized.append(distr)
    cluster = np.zeros(len(distributions)).astype(int)
    CCM = np.zeros((k1,k1))
    check = False
    counter = 0
    while check == False and counter <= 1000:
        check = True
        print('check')
        start = rand.sample(range(len(distributions)), k1)
        for i in range(len(start)):
            for j in range(i + 1, len(start)):
                temp = wasserstein_dist(distributions_normalized[start[i]],distributions_normalized[start[j]],CM)
                CCM[i][j] = temp
                if temp == 0:
                    check = False
                    counter += 1
        if check == True:
            center = start
    assert (check == True), "Initial centers with distance > 0 could not be selected!"
    print('----------------------------------------------------------------------------------------------------')
    print('Image Clustering running:')
    move_counter = 10**4
    i = 0
    # for i in range(stop):
    while i < stop and move_counter > 0:
        time1 = time.time()
        move_counter = 0
        for j in range(len(distributions)):
            initial = True
            temp1 = 0
            cluster_start = cluster[j]
            for k in range(len(center)):
                if initial:
                    temp1 = wasserstein_dist(distributions_normalized[j], distributions_normalized[center[k]], CM)
                    cluster[j] = k
                else:
                    temp2 = wasserstein_dist(distributions_normalized[j], distributions_normalized[center[k]], CM)
                    if temp2 < temp1:
                        cluster[j] = k
                        temp1 = temp2
                initial = False
            if cluster_start != cluster[j]:
                move_counter += 1
        time2 = time.time()
        print('Assignment to centroids done:\t' + str(time2 - time1) + '\ts.')
        time3 = time.time()
        for j in range(k1):
            initial = True
            indices = np.where(cluster == j)
            tmp_distr = distributions_normalized[indices[0][0]]
            for index in range(1,len(indices[0])):
                tmp_distr = np.vstack((tmp_distr, distributions_normalized[indices[0][index]]))
            tmp_distr = tmp_distr.T
            barycenter = ws_barycenter(tmp_distr, CM)
            tmp_distr = tmp_distr.T
            min_index = 10**9
            min_dist = 10**9
            for index in indices[0]:
                tmp_dist = wasserstein_dist(barycenter, distributions_normalized[index], CM)
                if tmp_dist < min_dist:
                    min_dist = tmp_dist
                    min_index = index
            center[j] = min_index
        time4 = time.time()
        print('Calculation of centroids done:\t' + str(time4 - time3) + '\ts.')
        i += 1
    print('----------------------------------------------------------------------------------------------------')
    print()
    return cluster, center


# ------Cloudmetrics Computation CMASK--------------------------------------------------------------------------------------
def compute_cloudmetrics_cmask(sample, images, cluster_images, center_images, name):
    print('----------------------------------------------------------------------------------------------------')
    print('Cloudmetrics Computation running.')
    fieldnames = ['path', 'cluster', 'center', 'fraction', 'mean length scale', 'max length scale', 'iOrg']
    rows = []
    counter = 0
    tmp5 = -1
    for i,j,l in zip(sample, images, cluster_images):
        for k in center_images:
            if counter == k:
                tmp5 = center_images.index(k)
        labels = cloudmetrics.objects.label(mask=j)
        try:
            tmp1 = cloudmetrics.mask.cloud_fraction(j)
        except:
            tmp1 = 0.0
        try: 
            tmp2 = cloudmetrics.objects.mean_length_scale(labels)
        except:
            tmp2 = 0.0
        try: 
            tmp3 = cloudmetrics.objects.max_length_scale(labels)
        except:
            tmp3 = 0.0
        try:
            tmp4 = cloudmetrics.objects.metrics.iorg(labels)
        except:
            tmp4 = 0.0
        row = {'path': i, 'cluster': l + 1, 'center': tmp5 + 1, 'fraction': tmp1, 'mean length scale': tmp2, 'max length scale': tmp3, 'iOrg': tmp4}
        rows.append(row)
        tmp5 = -1
        counter += 1
    print('Cloudmetrics successfully computed.')
    print('----------------------------------------------------------------------------------------------------')
    print()
    print('----------------------------------------------------------------------------------------------------')
    print('Writing results to csv.')
    with open('CM_metrics_' + name + '.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print('File saved.')
    print('----------------------------------------------------------------------------------------------------')
    print()


# ------Cloudmetrics Computation COT----------------------------------------------------------------------------------------
def compute_cloudmetrics_cot(sample, images, cluster_images, center_images, name, cod_tresh):
    print('----------------------------------------------------------------------------------------------------')
    print('Cloudmetrics Computation running.')
    fieldnames = ['path', 'cluster', 'center', 'fraction', 'mean length scale', 'max length scale', 'iOrg', 'cot']
    rows = []
    counter = 0
    tmp6 = -1
    for i,j,l in zip(sample, images, cluster_images):
        for k in center_images:
            if counter == k:
                tmp6 = center_images.index(k)
        cotmask = np.where(np.array(j) > cod_tresh, 1, 0)
        labels = cloudmetrics.objects.label(mask=cotmask)
        try:
            tmp1 = cloudmetrics.mask.cloud_fraction(cotmask)
        except:
            tmp1 = 0.0
        try: 
            tmp2 = cloudmetrics.objects.mean_length_scale(labels)
        except:
            tmp2 = 0.0
        try: 
            tmp3 = cloudmetrics.objects.max_length_scale(labels)
        except:
            tmp3 = 0.0
        try:
            tmp4 = cloudmetrics.objects.metrics.iorg(labels)
        except:
            tmp4 = 0.0
        try: 
            tmp5 = np.sum(j) / 128**2
        except:
            tmp5 = 0
        row = {'path': i, 'cluster': l + 1, 'center': tmp6 + 1, 'fraction': tmp1, 'mean length scale': tmp2, 'max length scale': tmp3, 'iOrg': tmp4, 'cot': tmp5}
        rows.append(row)
        tmp6 = -1
        counter += 1
    print('Cloudmetrics successfully computed.')
    print('----------------------------------------------------------------------------------------------------')
    print()
    print('----------------------------------------------------------------------------------------------------')
    print('Writing results to csv.')
    with open('COT_metrics_' + name + '.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print('File saved.')
    print('----------------------------------------------------------------------------------------------------')
    print()



# ------CMASK Clustering----------------------------------------------------------------------------------------------------
def cmask_clustering(sample_size, k_images, d2_stop, k_patches, patchsize, stepsize, tresh_patch_dist):
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
                time1 = time.time()
                heads_CM, distributions, cluster_patches = gonzalez_patches(patches, j, tresh_patch_dist)
                time2 = time.time()
                cluster_images, center_images = d2_images(k_images,distributions,heads_CM,d2_stop)
                time3 = time.time()
                compute_cloudmetrics_cmask(sample, images, cluster_images, center_images, name)
                time4 = time.time()
                print('Patch Clustering in ' + str(time2 - time1) + ' s.')
                print('Image Clustering in ' + str(time3 - time2) + ' s.')
                print('Computation of cloudmetrics in ' + str(time4 - time3) + ' s.')


# ------CMASK Clustering----------------------------------------------------------------------------------------------------
def cmask_clustering_fast(sample_size, k_images, d2_stop, k_patches, patchsize, stepsize, tresh_patch_dist):
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
                name = 'sample_size' + str(sample_size) + '_k_images' + str(k_images) + '_k_patches' + str(j) + '_patchsize' + str(i) + '_stepsize' + str(m)
                time1 = time.time()
                heads_CM, distributions, cluster_patches = gonzalez_patches(patches, j, tresh_patch_dist)
                time2 = time.time()
                cluster_images, center_images = d2_images_fast(k_images,distributions,heads_CM,d2_stop)
                time3 = time.time()
                compute_cloudmetrics_cmask(sample, images, cluster_images, center_images, name)
                time4 = time.time()
                print('Patch Clustering in ' + str(time2 - time1) + ' s.')
                print('Image Clustering in ' + str(time3 - time2) + ' s.')
                print('Computation of cloudmetrics in ' + str(time4 - time3) + ' s.')

    
# ------COT Clustering------------------------------------------------------------------------------------------------------
def cot_clustering_fast(sample_size, k_images, d2_stop, k_patches, patchsize, stepsize, tresh_patch_dist, tresh_cm_cot):
    filenames = glob.glob("/Users/sz/Documents/Uni/Master/Masterarbeit/Masterarbeit_Projekt/ImageClusteringMain/data/cot/random_10k_cot/*.npy")
    filenames.sort()
    sample = random.sample(filenames, sample_size)
    # images_original = []
    # for img in sample:
    #     images_original.append(np.load(img))
    images_tmp = []
    for img in sample:
        images_tmp.append(np.log(np.load(img) + 1))
    glob_min = []
    glob_max = []
    for img in images_tmp:
        glob_min.append(np.amin(img))
        glob_max.append(np.amax(img))
    images = [np.divide(img, max(glob_max), dtype=float) for img in images_tmp]
    # ana.plot_images_treshs(images_original, images)
    for i in patchsize:
        for j in k_patches:
            for m in stepsize[patchsize.index(i)]:
                patches = []
                for l in images:
                    patches.append(get_patches(l,i,m))
                name = 'sample_size' + str(sample_size) + '_k_images' + str(k_images) + '_k_patches' + str(j) + '_patchsize' + str(i) + '_stepsize' + str(m)
                time1 = time.time()
                heads_CM, distributions, cluster_patches = gonzalez_patches(patches, j, tresh_patch_dist)
                time2 = time.time()
                cluster_images, center_images = d2_images_fast(k_images,distributions,heads_CM,d2_stop)
                time3 = time.time()
                compute_cloudmetrics_cot(sample, images, cluster_images, center_images, name, tresh_cm_cot)
                time4 = time.time()
                print('Patch Clustering in ' + str(time2 - time1) + ' s.')
                print('Image Clustering in ' + str(time3 - time2) + ' s.')
                print('Computation of cloudmetrics in ' + str(time4 - time3) + ' s.')


