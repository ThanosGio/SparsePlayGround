import numpy as np
from compute_stat import compute_stat
from im2col import im2col
from col2im import col2im
from batch_thresholding import batch_thresholding
import matplotlib.pyplot as plt
#import MyTiler as MT
from omp import omp

def dct_image_denoising(noisy_im, D_DCT, epsilon):
    # DCT_IMAGE_DENOISING Denoise an image via the DCT transform
    # 
    # Inputs:
    #   noisy_im - The input noisy image
    #   D_DCT    - A column-normalized DCT dictionary
    #   epsilon  - The noise-level in a PATCH, 
    #              used as the stopping criterion of the pursuit
    #
    # Output:
    #  est_dct - The denoised image
    #
    
    # TODO: Get the patch size [height, width] from D_DCT
    # Write your code here... patch_size = ???
    patch_size = int(np.sqrt(D_DCT.shape[0]))

    # Divide the noisy image into fully overlapping patches
    patches = im2col(noisy_im, patch_size, stepsize=1)

    [est_patches, est_coeffs] = batch_thresholding(D_DCT, patches, epsilon)
    # [est_patches, est_coeffs] = omp(D_DCT, patches, 10, epsilon)
    # [est_patches, est_coeffs] = omp(D_DCT, patches, k, epsilon)

    # print("est patches shape", est_patches.shape)
    est_image_dct = col2im(est_patches, patch_size, noisy_im.shape)
    # est_image_dct = MT.patches2img(est_patches, noisy_im.shape[0], noisy_im.shape[1], step)

    compute_stat(est_patches, patches, est_coeffs)
 
    return  est_image_dct