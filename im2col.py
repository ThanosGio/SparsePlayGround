import numpy as np
# import cupy as onp

def im2col(A, patch_size, stepsize=1):
    # Parameters
    M,N = A.shape
    col_extent = N - patch_size + 1
    row_extent = M - patch_size + 1

    # Get Starting block indices
    start_idx = np.arange(patch_size)[:, None]*N + np.arange(patch_size)

    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent)[:, None]*N + np.arange(col_extent)

    # Get all actual indices & index into input array for final output
    return np.take(A, start_idx.ravel()[:, None] + offset_idx.ravel()[::stepsize])

def printSparseVector(x):
    print("sparse vector contents")
    for i in range(0,len(x)):
        if x[i] != 0:
            print("[", i,"]=", x[i])
