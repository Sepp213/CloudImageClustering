import numpy as np
from numba import njit
import image_processing as ip
import time

# ------Create random binary patches----------------------------------------------------------------------------------------
# Create random binary square images.
def get_rand_patch(dim):
    # Get random square data
    return np.random.random((dim, dim))



# ------Compute relabeling cost of one patch alignment without numba--------------------------------------------------------
def relabeling_costs(patch1, patch2):
    rc = 0
    # Compare patch1 and patch2 entrywise and add up absolute difference for deviating label
    for i in range(len(patch1)):
        for j in range(len(patch1[0])):
            rc += np.abs(np.subtract(patch1[i][j], patch2[i][j])) 
    return rc


# ------Compute relabeling cost of one patch alignment with numba-----------------------------------------------------------
@njit(fastmath=True)
def relabeling_costs_numba(patch1, patch2):
    rc = 0
    # Compare patch1 and patch2 entrywise and add up absolute difference for deviating label
    for i in range(len(patch1)):
        for j in range(len(patch1[0])):
            rc += np.abs(np.subtract(patch1[i][j], patch2[i][j]))    
    return rc



# ------Compute relabeling cost of one all patch alignments without numba---------------------------------------------------
# Method 1 for approximating GED.
def Delta_numpy(patch1, patch2):
    d = []
    # default
    d.append(relabeling_costs(patch1, patch2))
    # default flipped
    d.append(relabeling_costs(np.flipud(patch1), patch2))
    # rot 90
    d.append(relabeling_costs(np.rot90(patch1, 1), patch2))
    # rot 90 flipped
    d.append(relabeling_costs(np.flipud(np.rot90(patch1, 1)), patch2))
    # rot 180
    d.append(relabeling_costs(np.rot90(patch1, 2), patch2))
    # rot 180 flipped
    d.append(relabeling_costs(np.flipud(np.rot90(patch1, 2)), patch2))
    # rot 270
    d.append(relabeling_costs(np.rot90(patch1, 3), patch2))
    # rot 270 flipped
    d.append(relabeling_costs(np.flipud(np.rot90(patch1, 3)), patch2))
    return min(d)



# ------Compute relabeling cost of one all patch alignments with numba------------------------------------------------------
def Delta_numpy_numba(patch1, patch2):
    d = []
    # default
    d.append(relabeling_costs_numba(patch1, patch2))
    # default flipped
    d.append(relabeling_costs_numba(np.flipud(patch1), patch2))
    # rot 90
    d.append(relabeling_costs_numba(np.rot90(patch1, 1), patch2))
    # rot 90 flipped
    d.append(relabeling_costs_numba(np.flipud(np.rot90(patch1, 1)), patch2))
    # rot 180
    d.append(relabeling_costs_numba(np.rot90(patch1, 2), patch2))
    # rot 180 flipped
    d.append(relabeling_costs_numba(np.flipud(np.rot90(patch1, 2)), patch2))
    # rot 270
    d.append(relabeling_costs_numba(np.rot90(patch1, 3), patch2))
    # rot 270 flipped
    d.append(relabeling_costs_numba(np.flipud(np.rot90(patch1, 3)), patch2))
    return min(d)


patches_source = []
patches_target = []

for k in range(10**6):
    patches_source.append(get_rand_patch(8))
    patches_target.append(get_rand_patch(8))

time_start = time.time()
tmp = 0
for p1, p2 in zip(patches_source, patches_target):
    tmp += Delta_numpy(p1,p2)
time_end = time.time()
print('Delta_numpy: Total APED = ' + str(tmp) + ' computed in ' + str(time_end - time_start) + 's.')
print()

time_start = time.time()
tmp = 0
for p1, p2 in zip(patches_source, patches_target):
    tmp += Delta_numpy_numba(p1,p2)
time_end = time.time()
print('Delta_numpy_numba: Total APED = ' + str(tmp) + ' computed in ' + str(time_end - time_start) + 's.')
print()

time_start = time.time()
tmp = 0
for p1, p2 in zip(patches_source, patches_target):
    tmp += ip.Delta_final(p1,p2)
time_end = time.time()
print('Delta_final: Total APED = ' + str(tmp) + ' computed in ' + str(time_end - time_start) + 's.')
print()
