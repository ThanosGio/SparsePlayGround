import numpy as np

def col2im(patches, patch_size, im_size):
    # COL_TO_IM Rearrange matrix columns into an image of size MXN
    #
    # Inputs:
    #  patches - A matrix of size p * q, where p is the patch flatten size (height * width = m * n), and q is number of patches.
    #  patch_size - The size of the patch [height width] = [m n]
    #  im_size    - The size of the image we aim to build [height width] = [M N]
    #
    # Output:
    #  im - The reconstructed image, computed by returning the patches in
    #       'patches' to their original locations, followed by a
    #       patch-averaging over the overlaps

    num_im = np.zeros((im_size[0], im_size[1]))
    denom_im = np.zeros((im_size[0], im_size[1]))

    cnt = 0
    for i in range(0, im_size[0] - patch_size + 1, 1):
        for j in range(0, im_size[1] - patch_size + 1, 1):
            # rebuild current patch
            num_of_curr_patch = i * (im_size[1] - patch_size + 1) + (j + 1)

            last_row = i + patch_size
            last_col = j + patch_size
            curr_patch = patches[:, num_of_curr_patch - 1]
            curr_patch = np.reshape(curr_patch, (patch_size, patch_size))

            # update 'num_im' and 'denom_im' w.r.t. 'curr_patch'
            num_im[i:last_row, j:last_col] = num_im[i:last_row, j:last_col] + curr_patch
            denom_im[i:last_row, j:last_col] = denom_im[i:last_row, j:last_col] + np.ones(curr_patch.shape)

    # Averaging
    im = num_im / denom_im

    return im



def im2col_sliding_strided(A, BSZ, stepsize=1):
    # Parameters
    m,n = A.shape
    s0, s1 = A.strides
    nrows = m-BSZ[0]+1
    ncols = n-BSZ[1]+1
    shp = BSZ[0],BSZ[1],nrows,ncols
    strd = s0,s1,s0,s1

    out_view = np.lib.stride_tricks.as_strided(A, shape=shp, strides=strd)
    return out_view.reshape(BSZ[0]*BSZ[1],-1)[:,::stepsize]
