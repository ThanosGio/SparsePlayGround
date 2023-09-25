#CCH 20210409 DEMO 01: RESTORE IMAGE WITH TRAINED DCT DICTIONARY BY PROCRUSTES

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.util import view_as_windows as viewW
from skimage.util import view_as_blocks as viewB
from build_dct_unitary_dictionary import build_dct_unitary_dictionary
from show_dictionary import show_dictionary
from batch_thresholding import batch_thresholding
from compute_stat import compute_stat
from unitary_dictionary_learning import unitary_dictionary_learning

# CCH 20200520 widthen display output
import pandas as pd

desired_width = 260
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 10)
np.set_printoptions(linewidth=desired_width)
np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float': lambda x: " {0:7.3f}".format(x)})
#np.set_printoptions(threshold=25)
# Toon volledige matrices
#np.set_printoptions(threshold=np.inf)

# %% Part A: Data Construction and Parameter-Setting

# Read an image
im = np.array(Image.open('barbara.png'))
#im = np.array(Image.open('brain4.jpg'))
# im = np.array(Image.open('brain4_r01.jpg'))

# Show the image
plt.figure(0)
plt.imshow(im, 'gray')
plt.title('Original image')
# plt.show()

#Cardinaliteit van de sparse vectors
K = 8

# Patch dimensions [height, width]
dim = 8
patch_size = [dim, dim]



# Aantal iteraties om te leren
T=25

# Set the seed for the random generator
seed = 10827

# Set a fixed random seed to reproduce the results
np.random.seed(seed)

#TRAINING FROM HERE

# Create the overlapping patches for training
all_train_patches = viewW(im, patch_size)
# Number of patches to train on
num_train_patches = 20000

train_patches_idx = np.random.permutation((512 - dim) * (512 - dim))
train_patches = np.zeros((dim * dim, num_train_patches))

for i in range(num_train_patches):
    tp_x = np.floor_divide(train_patches_idx[i], 512 - dim)
    tp_y = np.mod(train_patches_idx[i], 512 - dim)
    train_patches[:, i] = all_train_patches[tp_x, tp_y].flatten()


# CCH 20210409 Maak en DCT dictionary
D_DCT = build_dct_unitary_dictionary(patch_size)

# Train a dictionary via Procrustes analysis
D_learned, mean_error, mean_cardinality = unitary_dictionary_learning(train_patches, D_DCT, T, K)
plt.figure(1)
# plt.subplot(1, 2, 2)
show_dictionary(D_learned)
plt.title("Learned Unitary Dictionary")
#plt.show()

# Compute the representation of each patch that belongs to the training set using Thresholding
est_train_patches_dct, est_train_coeffs_dct = batch_thresholding(D_learned, train_patches, K)
# print('coef:', est_train_coeffs_dct[:,0])

# Compute and display the statistics
print('\n\nDCT dictionary: Training set, ')
compute_stat(est_train_patches_dct, train_patches, est_train_coeffs_dct)

#TESTING FROM HERE
#CCH 20210409 64*64 bedekt het hele plaatje met niet-overlappende patches
all_test_patches = viewB(im, block_shape=(dim, dim))

# The whole image in non overlapping patches (64*64)
num_test_patches = 4096
#CCH 20210409 Initialiseer de patches met 0-waarden
test_patches = np.zeros((dim * dim, num_test_patches))

#CCH 20210409 Zet alle patches in een matrix van 64*4096
for i in range(num_test_patches):
    tp_x = np.floor_divide(i, dim*dim)
    tp_y = np.mod(i, dim*dim)
#    print('tpx: ', tp_x, ' tpy:', tp_y)
    test_patches[:, i] = all_test_patches[tp_x, tp_y].flatten()

#CCH 20210409 test_patches voeren aan de nieuw geleerde dictionary

est_test_patches_dct, est_test_coeffs_dct = batch_thresholding(D_learned, test_patches, K)

#CCH 20210409 En nu de est_test_patches_dct weer omsmurfen van 64*4096 naar 512*512 plaatje

restored_image = np.zeros((512, 512))

# i iterates from 0 to num_train_patches-1
for i in range(num_test_patches):
    tp_x = np.floor_divide(i, dim*dim)
    tp_y = np.mod(i, dim*dim)
#    print('i: ', i, ' tpx: ', tp_x, ' tpy:', tp_y)
    pt = est_test_patches_dct[:, i].reshape(8, 8)
    restored_image[tp_x*dim:tp_x*dim+pt.shape[0], tp_y*dim:tp_y*dim+pt.shape[1]] += pt

plt.figure(5)
plt.imshow(restored_image, 'gray')
plt.title('Restored image (K=' + str(K) + ')')

# Compute and display the statistics
print('\nDCT dictionary: Test set, ')
compute_stat(est_test_patches_dct, test_patches, est_test_coeffs_dct)

# # Show the representation error and the cardinality as a function of the learning iterations
# plt.figure(2)
# plt.subplot(1, 2, 1)
# plt.plot(np.arange(T), mean_error, linewidth=2.0)
# plt.ylabel("Average Representation Error")
# plt.xlabel("Learning Iteration")
# plt.subplot(1, 2, 2)
# plt.plot(np.arange(T), mean_cardinality, linewidth=2.0)
# plt.ylabel('Average Number of Non-Zeros')
# plt.ylim((K-1, K+1))
# plt.xlabel('Learning Iteration')

plt.show()