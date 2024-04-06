from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        lg_c = -scores.max()
        expd = np.exp(scores + lg_c)
        softmax = expd / expd.sum()
        loss += -np.log(softmax[y[i]])

        dsftmx = -1 / softmax[y[i]]
        dexpd = dsftmx * softmax[y[i]] * ((np.arange(num_classes) == y[i]) - softmax)
        dW += np.outer(X[i], dexpd)

    loss /= num_train
    loss += reg * (W * W).sum()

    dW /= num_train
    dW += 2 * reg * W 

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]
    scores = X @ W
    expd = np.exp(scores - scores.max(axis=1, keepdims=True))
    softmax = expd / expd.sum(axis=1, keepdims=True)
    like = softmax[np.arange(num_train), y]
    loss = np.mean(-np.log(like)) + reg * np.sum(W ** 2)

    # gradient using backprop. can be easier dexpd = -del_j_yi + expd
    dsftmx = -1 / like
    del_ij = np.zeros_like(softmax)
    del_ij[np.arange(num_train), y] = 1
    dexpd = (dsftmx * softmax[np.arange(num_train), y])[:,np.newaxis] * (del_ij - softmax)
    dW = X.T @ dexpd / num_train + 2 * reg * W


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
