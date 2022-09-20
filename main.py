from pickletools import uint8
from keras.datasets import mnist
import numpy as np
import image_processing as ip
import ot 
import edh_ged as edh
import cv2 as cv
import glob
import random
import time

def main():

    # ------MNIST Testing---------------------------------------------------------------------------------------------------
    # (train_X, train_Y), (test_X, test_Y) = mnist.load_data()        # Load MNIST dataset
    # images = []                                           # List to store the images  
    # l = 1000                                              # Number of images 
    # k1 = 10                                               # Number of image clusters
    # k2 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]        # Number of patch clusters
    # patchsize = [2, 3, 4, 6, 8, 10, 12, 14, 16]           # Size of patches
    
    # counter0 = 0
    # counter1 = 0
    # counter2 = 0
    # counter3 = 0
    # counter4 = 0
    # counter5 = 0
    # counter6 = 0
    # counter7 = 0
    # counter8 = 0
    # counter9 = 0
    
    # for i in range(len(test_X)):        # Append l/10 of each digit to the image list
    #     if test_Y[i] == 0 and counter0 < l/10:
    #         images.append(np.where(test_X[i] > 0, 1, 0).astype(np.uint8)) 
    #         counter0 += 1
    #     if test_Y[i] == 1 and counter1 < l/10:
    #         images.append(np.where(test_X[i] > 0, 1, 0).astype(np.uint8)) 
    #         counter1 += 1
    #     if test_Y[i] == 2 and counter2 < l/10:
    #         images.append(np.where(test_X[i] > 0, 1, 0).astype(np.uint8)) 
    #         counter2 += 1
    #     if test_Y[i] == 3 and counter3 < l/10:
    #         images.append(np.where(test_X[i] > 0, 1, 0).astype(np.uint8)) 
    #         counter3 += 1
    #     if test_Y[i] == 4 and counter4 < l/10:
    #         images.append(np.where(test_X[i] > 0, 1, 0).astype(np.uint8)) 
    #         counter4 += 1
    #     if test_Y[i] == 5 and counter5 < l/10:
    #         images.append(np.where(test_X[i] > 0, 1, 0).astype(np.uint8)) 
    #         counter5 += 1
    #     if test_Y[i] == 6 and counter6 < l/10:
    #         images.append(np.where(test_X[i] > 0, 1, 0).astype(np.uint8)) 
    #         counter6 += 1
    #     if test_Y[i] == 7 and counter7 < l/10:
    #         images.append(np.where(test_X[i] > 0, 1, 0).astype(np.uint8)) 
    #         counter7 += 1
    #     if test_Y[i] == 8 and counter8 < l/10:
    #         images.append(np.where(test_X[i] > 0, 1, 0).astype(np.uint8)) 
    #         counter8 += 1
    #     if test_Y[i] == 9 and counter9 < l/10:
    #         images.append(np.where(test_X[i] > 0, 1, 0).astype(np.uint8)) 
    #         counter9 += 1
    
    
    # for a in k2:
    #     for b in patchsize:  
    #         name = 'D2_Clustering_k2' + str(a) + '_patchsize' + str(b)  + '_stop10'          
    #         patches = [] 
    #         for i in images:
    #             patches.append(ip.get_patches(i,b))
    #         heads, heads_DM, distributions, cluster = ip.gonzalez_patches(patches, a, False)
    #         cluster_images, center_images = ip.d2_images(k1, distributions, heads_DM, 10)
    #         ip.plot_clustering(images, cluster_images, 0, 10, 0, name)
    # ----------------------------------------------------------------------------------------------------------------------

    # ------CMASK Testing---------------------------------------------------------------------------------------------------
    path = "/Users/sz/Documents/Uni/Master/Masterarbeit/Masterarbeit_Projekt/ImageClusteringMain/data/cmask/germany_cmask_128x128_npy/"
    filenames = glob.glob("/Users/sz/Documents/Uni/Master/Masterarbeit/Masterarbeit_Projekt/ImageClusteringMain/data/cmask/germany_cmask_128x128_npy/*.npy")
    filenames.sort()
    load = [np.load(arr) for arr in filenames]
    sample = random.sample(load, 1000)
    images = []
    for i in sample:
        images.append(i.astype(int))
    k_images = 8
    k_patches = 3
    bool_rot = True
    patchsize = 100
    patches = []
    # for i in images:
    #     patches.append(ip.get_patches(i, patchsize))
    # heads, heads_CM, distributions, cluster_patches = ip.gonzalez_patches(patches, k_patches, bool_rot)
    # cluster_images, center_images = ip.d2_images(k_images, distributions, heads_CM, 20)
    # ip.plot_clustering(images, cluster_images, 0, k_images, 0, "TEST")
    start = time.time()
    print(ip.app_ged(images[0], images[1], True))
    end = time.time()
    print(end - start)
    start = time.time()
    print(ip.app_ged(images[0], images[1], True))
    end = time.time()
    print(end - start)
        
# Run main
if __name__ == "__main__":
    main()