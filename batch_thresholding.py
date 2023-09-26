import numpy as np

def batch_thresholding(D, Y, K):
    
    # BATCH_THRESHOLDING Solve the pursuit problem via Thresholding using 
    # a fixed cardinality as the stopping criterion.
    # 
    # Solves the following problem:
    #   min_{alpha_i} \sum_i ||y_i - D alpha_i||_2^2 
    #                   s.t. || alpha_i ||_0 = K for all i,
    # where D is a dictionary of size n X n, y_i are the input signals of
    # length n, and K stands for the number of nonzeros allows per each alpha_i
    # 
    # The above can be equally written as follows:
    #   min_{A} \sum_i ||Y - DA||_F^2 
    #             s.t. || alpha_i ||_0 = K for all i,
    # where Y is a matrix that contains the y_i patches as its columns.
    # Similarly, and the matrix A contains the representations alpha_i 
    # as its columns
    #
    # The solution is returned in the matrix A, along with the denoised signals
    # given by  X = DA.

    # TODO: Compute the inner products between the dictionary atoms and the
    # input patches (hint: the result should be a matrix of size n X N)
    # Write your code here... inner_products = ???

    inner_products =np.dot(D.transpose(), Y)
    # CCH 20210212 Dat kostte wel een poosje om vast te stellen dat hier Dt genomen moet worden ipv D
    # CCH 20210212 Dat komt omdat je de uitleg niet goed leest Coen, daar staat het gewoon netjes in :-)

    # print('inner shape', inner_products.shape)
    # print('inner products:', inner_products)

    # TODO: Compute the absolute value of the inner products
    # Write your code here... abs_inner_products = ???
    abs_inner_products =np.abs(inner_products)

    # TODO: Sort the absolute value of the inner products in a descend order.
    # Notice that one can sort each column independently
    # Write your code here... mat_sorted = np.sort(?, ?)
    mat_sorted = np.sort(-abs_inner_products,  axis=0)
    mat_sorted = -mat_sorted

    #print('math sorted:')
    #print(mat_sorted[:,0])
    #print(mat_sorted[:,1])

    # TODO: Compute the threshold value per patch, by picking the K largest entry in each column of 'mat_sorted'
    # Write your code here... vec_thresh = mat_sorted[?,?]
    vec_thresh = mat_sorted[K-1]
    # print('vec_tresh', vec_thresh)

    # Replicate the vector of thresholds and create a matrix of the same size as 'inner_products'
    mat_thresh = np.tile(np.expand_dims(vec_thresh,axis=0),(np.shape(abs_inner_products)[0],1))
    #print('mat_tresh_shape: ', mat_thresh.shape)
    #print('mat thresh', mat_thresh)

    # TODO: Given the thresholds, initialize the matrix of coeffecients to be equal to 'inner_products' matrix
    # Write your code here... A = ???
    A = inner_products
    # print('AIP0:',A[:,0])

    # TODO: Set the entries in A that are smaller than 'mat_thresh' (in absolute value) to zero
    # CCH 20210129 Some hard thresholding going on here...
    # Write your code here... A[ ??? ] = 0

    # CCH 20210212 This is why I love Python:
    A = (abs(A)>=mat_thresh) * A

    # print('A0:',A[:,0])
    #CCH 20210205 Maar waar gaan we nu de coeficienten van A berekenen? Dus dictonairy patch * coefficient = Benadering van de patch.
    #CCH 20210205 Uit sparserep1 cursus blijkt dat bij benadering het improduct mag worden gezien als de oplossing van de least squares...

    #TODO: Compute the estimated patches given their representations in 'A'
    # Write your code here... X = ???
    X = np.dot(D, A)

    # print('X shape:', X.shape)
    # print(X)

    return X, A
