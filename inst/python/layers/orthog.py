import tensorflow as tf
import numpy as np

def orthog_col(U, X):
    
    C = tf.matmul(tf.transpose(U), X)
    ncolU = U.shape[1]
    ncolX = X.shape[1]
    ncolMax = np.max([ncolU, ncolX])
    padrow = ncolMax - ncolU
    padcol = ncolMax - ncolX
    paddings = tf.constant([[0, padrow], [0, padcol]])
    C = tf.pad(C, paddings)
    s, Q, _ = tf.linalg.svd(C, full_matrices=True)
#     Q = tf.linalg.qr(C, full_matrices=True, name="QR").q
    rank_C = s.shape[0] #tf.linalg.matrix_rank(C)
    mulpl = tf.concat([tf.zeros((ncolMax, rank_C)), 
                       tf.ones((ncolMax, ncolMax-rank_C))], 
                      axis=1)
    Z = tf.multiply(mulpl, tf.cast(Q, tf.float32))
    return(tf.matmul(tf.cast(U, tf.float32), Z[0:U.shape[1],0:U.shape[1]]))
