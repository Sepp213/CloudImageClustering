from cmath import pi
from distutils.dist import DistributionMetadata
import image_processing as ip
import numpy as np
import cv2 as cv
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import math

def corner_detector(image):
    
    # tmp = image 
    rgb = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    # gray = cv.cvtColor(rgb, cv.COLOR_BGR2GRAY)
    gray = np.float32(image)
    
    points = []
    dst = cv.cornerHarris(gray, 2, 3, 0.04)
    dst = cv.dilate(dst,None)
    for i in range(len(dst)):
        for j in range(len(dst[0])):
            if dst[i][j]>0.01*dst.max():
                points.append([j, len(dst) - i])
    
    rgb[dst>0.01*dst.max()]=[0,0,255]
    
    # cv.imshow('dst',rgb)
    # if cv.waitKey(0) & 0xff == 27:
    #     cv.destroyAllWindows()
    
    return np.array(points)

def delaunay(points):
    
    tri = Delaunay(points)
 
    # plt.triplot(points[:,0], points[:,1], tri.simplices)
    # plt.plot(points[:,0], points[:,1], 'o')
    # plt.show()

    return tri.simplices


def all_angles(points, triangles):
    angles = []
    lengths = []
    for i in range(len(triangles)):
        angles.append(int(angle(points[triangles[i][0]] - points[triangles[i][1]])))
        lengths.append(np.linalg.norm(points[triangles[i][0]] - points[triangles[i][1]]))
        angles.append(int(angle(points[triangles[i][0]] - points[triangles[i][2]])))
        lengths.append(np.linalg.norm(points[triangles[i][0]] - points[triangles[i][2]]))
        angles.append(int(angle(points[triangles[i][1]] - points[triangles[i][2]])))
        lengths.append(np.linalg.norm(points[triangles[i][1]] - points[triangles[i][2]]))
    return angles, lengths


def angle(v1):
    cos_alpha = v1[1] / np.linalg.norm(v1)
    alpha = np.arccos(cos_alpha)
    return math.degrees(alpha)


def compute_distribution(angles, lengths):
    distribution = np.zeros(180) # --> 4 to 180
    for i in angles:
    #     if i > 7/8*np.pi or i <= np.pi/8:
    #         distribution[0] += 1 * lengths[angles.index(i)]
    #     elif i > np.pi/8 and i <= 3/8*np.pi:
    #         distribution[1] += 1 * lengths[angles.index(i)]
    #     elif i > 3/8*np.pi and i <= 5/8*np.pi:
    #         distribution[2] += 1 * lengths[angles.index(i)]
    #     elif i > 5/8*np.pi and i <= 7/8*np.pi:
    #         distribution[3] += 1 * lengths[angles.index(i)]
 
        distribution[i - 1] += 1 * lengths[angles.index(i)]
    norm = sum(distribution)
    
    return distribution / norm


def all_distributions(images):
    distributions = []
    for i in images:
        points = corner_detector(i)
        triangles = delaunay(points)
        angles, lengths = all_angles(points, triangles)
        distributions.append(compute_distribution(angles, lengths))
    return distributions


def ed_CM():
    CM = np.array([[0, 1/8*np.pi, 1/4*np.pi, 1/8*np.pi], [1/8*np.pi, 0, 1/8*np.pi, 1/4*np.pi], [1/4*np.pi, 1/8*np.pi, 0, 1/8*np.pi], [1/8*np.pi, 1/4*np.pi, 1/8*np.pi, 0]])
    return CM

def ed_CM2():
    CM = np.zeros((180,180))
    for i in range(180):
        for j in range(180):
            CM[i][j] = min([abs(i - j), abs(j - i)])
    return CM

def resize(image, x, y):
    
    res = cv.resize(image, dsize=(x, y), interpolation=cv.INTER_NEAREST)
    
    return res


def d2_images(k1, distributions, CM, stop):
    
    cluster = np.zeros(len(distributions))
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
        for j in range(len(distributions)):
            print('Iteration ' + str(i) + ':\tAssignment\t' + str(j + 1) + '\tof\t' + str(len(distributions)) + '.')
            initial = True
            temp1 = 0
            for k in range(len(center)):
                if initial:
                    temp1 = ip.wasserstein_dist(distributions[j], distributions[center[k]], CM)
                    cluster[j] = k
                    initial = False
                else:
                    temp2 = ip.wasserstein_dist(distributions[j], distributions[center[k]], CM)
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
                #counter = 1
                for l in indices[0]:
                    #print('Calculation of Centroid ' + str(j + 1) + ': ' + str(counter) + ' of ' + str(len(indices[0]) * len(indices[0])))
                    #counter += 1
                    temp += ip.wasserstein_dist(distributions[k], distributions[l], CM)
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
