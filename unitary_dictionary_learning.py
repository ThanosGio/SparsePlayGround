import numpy as np
from compute_stat import compute_stat
from batch_thresholding import batch_thresholding
import matplotlib.pyplot as plt

def unitary_dictionary_learning(Y, D_init, num_iterations, pursuit_param):
    
    # UNITARY_DICTIONARY_LEARNING Train a unitary dictionary via 
    # Procrustes analysis.
    #
    # Inputs:
    #   Y              - A matrix that contains the training patches 
    #                    (as vectors) as its columns
    #   D_init         - Initial UNITARY dictionary
    #   num_iterations - Number of dictionary updates
    #   pursuit_param  - The stopping criterion for the pursuit algorithm
    #
    # Outputs:
    #   D          - The trained UNITARY dictionary
    #   mean_error - A vector, containing the average representation error,
    #                computed per iteration and averaged over the total 
    #                training examples
    #   mean_cardinality - A vector, containing the average number of nonzeros,
    #                      computed per iteration and averaged over the total 
    #                      training examples
   
    # Allocate a vector that stores the average representation error per iteration
    mean_error = np.zeros(num_iterations)

    # Allocate a vector that stores the average cardinality per iteration
    mean_cardinality = np.zeros(num_iterations)
 
    # TODO: Set the dictionary to be D_init
    # Write your code here... D = ???
    D = D_init
    print('Y shape: ', Y.shape)

    #plt.figure(1)
    #plt.subplot(1, 2, 2)


    # Run the Procrustes analysis algorithm for num_iterations
    #for iter = 1 : num_iterations
    for i in range(num_iterations):
        # plt.figure(5+i)
        # print(D)
        # Compute the representation of each noisy patch
        [X, A] = batch_thresholding(D, Y, pursuit_param)

        # CCH 20210212 X = gegenereerde patch, A = coefficient matrix, D = Dictionary, Y = training patches

        # Compute and display the statistics
        print('Iter %2d: ' % i)
        mean_error[i], mean_cardinality[i] = compute_stat(X, Y, A)
 
        # TODO: Update the dictionary via Procrustes analysis.
        # Solve D = argmin_D || Y - DA ||_F^2 s.t. D'D = I,
        # where 'A' is a matrix that contains all the estimated coefficients,
        # and 'Y' contains the training examples. Use the Procrustes algorithm.
        # Write your code here... D = ???

        AYt = np.dot(A, Y.transpose())

        # print('AYt.shape: ', AYt.shape)
        # dit is 36x36

        U, S, Vt = np.linalg.svd(AYt, full_matrices=False)

        # CCH 20210212 Because Vt is returned by SVD we have to transpose it again to get the original V matrix
        D = np.dot(Vt.transpose(), U.transpose())

        # show_dictionary(D)
        # plt.show()
        # print('D shape: ', D.shape)
    return D, mean_error, mean_cardinality