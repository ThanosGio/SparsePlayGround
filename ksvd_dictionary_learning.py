import numpy as np
from compute_stat import compute_stat
from omp import omp
import time
import show_dictionary as showdict

from numba import jit

@jit(nopython=True)
def ksvd_dictionary_learning(Y, D_start, num_iterations, k, epsilon):
    # CCH 20210709
    # K-SVD dictionary learning

    # Steps in the algorithm
    #   * A Run a solver (OMP/Thresholding) to compute an A vector
    #   * B Take the first dictionary atom and look for the patches that use it
    #   * C Create a vector that contains the contribution of this single atom
    #       * what about the error
    #   * D Run SVD on this 'residu'
    #   * E Assign S, V and D to the atom, the patches which use it and A
    #       * Formula found in matlab sample code
    #   * F Assign SVD to those elements
    #   * G Start from the beginning (solver)
    #   * H After all dictionary atoms are treated, repeat the proces <num_iterations> times


    D = D_start
    mean_error = np.zeros(num_iterations)

    # Allocate a vector that stores the average cardinality per iteration
    mean_cardinality = np.zeros(num_iterations)

    # D_prev = np.ndarray.copy(D)
    # print('Y.shape:', Y.shape)
    # Y = Y[:, 10000:35000]

    # Run the KSVD algorithm for num_iterations
    for i in range(num_iterations):
        print('-------K-SVD iteration', i+1) #speciaal voor Hetty :-)
        # time.sleep(1)

        # Step A
        # [X, A] = batch_thresholding(D, Y, epsilon*1.1)
        print('omp start')
        # t1 = time.time()
        [X, A] = omp(D, Y, k, epsilon)
        # print('omp finish {:.2f}s'.format(time.time() - t1))

        # Compute and display the statistics
        mean_error[i], mean_cardinality[i] = compute_stat(X, Y, A)
        # print('error / cardinality: {:.2f}'.format(mean_error[i]) + ' {:.2f}'.format(mean_cardinality[i]))
        # time.sleep(1)

        # CCH 20210713 Met KSVD gaan we alle atoms bij langs
        # t1 = time.time()
        print('KSVD iteration start')
        for j in range(D.shape[1]):
            idx = np.nonzero(A[j, :])[0]
            # print('idx len:', len(idx))

            # Controleer of het atom wordt gebruikt
            if (len(idx)>0):
                # Dj_oud = np.ndarray.copy(D[:, j])
                # D[:, j] = 0
                A[j] = 0.0
                residu = (Y - D@A)[:, idx]
                # print('resudi shape:', residu.shape)
                # print('residu (zonder Dj):', residu)
                # print('energy:', np.linalg.norm(residu))
                #            print('norm residu:', np.linalg.norm(residu))
                U, S, Vt = np.linalg.svd(residu, full_matrices=False)

                # print('shapes:', U.shape, S.shape, Vt.shape)
                D[:, j] = U[:, 0]
                # print('nieuwe atom :', U[:, 0])
                for k1 in range(len(idx)):
                    A[j, idx[k1]] = S[0] * Vt[0, k1]

        # print('KSVD iteration finished {:.2f}s'.format(time.time() - t1))

    print('KSVD learning finished')
    return D, mean_error, mean_cardinality

