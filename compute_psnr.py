import numpy as np
import math

def compute_psnr(y_original, y_estimated):
    
    # COMPUTE_PSNR Computes the PSNR between two images
    #
    # Input:
    #  y_original  - The original image
    #  y_estimated - The estimated image
    #
    # Output:
    #  psnr_val - The Peak Signal to Noise Ratio (PSNR) score

    y_original = np.reshape(y_original,(-1))
    y_estimated = np.reshape(y_estimated,(-1))

    dynamic_range = 255.0

    mse_val = (1/len(y_original)) * np.sum((y_original - y_estimated)**2)

    if mse_val == 0:
        psnr_val = 100
    else:
        psnr_val = 10*math.log10(dynamic_range**2 / mse_val)

    return psnr_val

#CCH 20210917 gelijk aan de eerste, maar wat compacter
def compute_psnr2(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    print('mse2: ', mse)
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr