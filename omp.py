import numpy as np
from tqdm import tqdm
from numba import jit, cuda


@jit(nopython=True)

def omp(D, X, k, epsilon):
    # OMP Solve the P0 problem via OMP
    #
    # Solves the following problem:
    #   min_x ||X - D.A||_2^2 s.t. ||A||_0 <= k
    #
    # The solution is returned in the vector A

    # CCH 20211221 convert INT X to FLOAT X for correct processing in JIT
    X_float = X + 1E-5

    A = np.zeros((np.shape(X)[1], D.shape[0]))

    sup = np.zeros(D.shape[0])
    X_solved = np.zeros(X.shape)

    max_steps = 0
    sum_steps = 0

    # CCH 20210713 traverse all patches
    for p in range(np.shape(X)[1]):

        # for p in range(1):
        # print('numpy in OMP now: ', p)
        A_calc = np.zeros(D.shape[0])
        norm_residu = np.linalg.norm(X_float[:,p])
        if norm_residu == 0:
            norm_residu = 1

        # CCH Initialize residual (met gehele signaal)
        residu_ca = np.ascontiguousarray(X_float[:,p])
        X_mat = residu_ca / norm_residu

        step = 0

        # print(np.linalg.norm(residu), epsilon)

        A0 = np.zeros((20, D.shape[0]))

        while ((np.linalg.norm( residu_ca ) > epsilon) and (step < k)):

            # print("p, STEP, norm residu, epsilon", p , step, np.linalg.norm(residu), epsilon)
            # printSparseVector(residu)
            v = np.dot(D.T, residu_ca)

            # print(v.shape)
            # CCH get max inner product column = best fitting atom from D
            A_idx = np.argmax(np.abs(v))
            # print( 'A idx:', A_idx)
            sup[step] = A_idx

            A0[step] = D[:, A_idx]

            # c = np.linalg.lstsq(A0[:step+1].T, X_mat.T, rcond=None)
            c = np.linalg.lstsq(A0[:step + 1].T, X_mat.T)

            # CCH update A
            for i in range(0, step+1):
                #A_calc[np.int(sup[i].item())] = c[0][i].item()
                A_calc[int(sup[i].item())] = c[0][i].item()


            # CCH 20200504 residu berekenen
            residu_ca = np.ascontiguousarray(np.asarray(np.subtract(X_float[:, p] / norm_residu, np.dot(D, A_calc))))

            step = step + 1

        if step == k:
            max_steps += 1

        A[p,:] = A_calc

        X_solved[:,p] = np.dot(D, A[p,:].T)
        X_solved[:,p] = X_solved[:,p] * norm_residu

    # print('OMP max step quit % {:.2f}'.format(100 * max_steps / np.shape(X)[1]))

    return X_solved , A.T

def printSparseVector(x):
    print("sparse vector contents")
    for i in range(0,len(x)):
        if x[i] != 0:
            print("[", i,"]=", x[i])