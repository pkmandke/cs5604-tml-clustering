'''

Common utilities

@pkmandke

'''


import os
import time
from datetime import timedelta

from tqdm import tqdm
import numpy as np

def compute_optimal_k_dists(data, k, num_pts=-1):

    if num_pts == -1:
        num_pts = data.shape[0]
        print(num_pts)

    k_dist = []
    
    t1 = time.monotonic()
    for point in range(0, num_pts):
        
        print("Computing {0}-dist for {1}th point".format(k, point + 1), end='\r')
        k_dist.append(sorted([np.linalg.norm(data[ix, :] - data[point, :]) for ix in range(num_pts) if ix != point])[k-1])
    print()
    print("Time taken {}s".format(timedelta(seconds=time.monotonic() - t1)))

    return k_dist