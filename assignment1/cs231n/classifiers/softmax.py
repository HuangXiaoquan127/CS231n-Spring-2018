import numpy as np
from random import shuffle

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
  N = X.shape[0]
  D, C = W.shape[0], W.shape[1]
  for i in range(N) :
      z = X[i].dot(W)
      loss += -np.log(np.exp(z[y[i]]) / np.sum(np.exp(z)))
    
      ds = np.exp(z) / np.sum(np.exp(z))
      ds[y[i]] += -1
      dW += X[i].reshape(D,1).dot(ds.reshape(1,C))
      
      
  loss /= N
  loss += reg * np.sum(W * W)
  
  dW /= N
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  N = X.shape[0]

  Z = X.dot(W)
  Z -= np.amax(Z, axis=1, keepdims=True)
  Z_exp = np.exp(Z)
  Z_hsum = np.sum(Z_exp, axis = 1, keepdims=True)
  loss = np.mean(-np.log(Z_exp[np.arange(N), y] / Z_hsum))
  loss += reg * np.sum(W * W)
    
  dZ = Z_exp / Z_hsum
  dZ[np.arange(N), y] += -1
  dW = X.T.dot(dZ)
  dW /= N
  dW += 2 * reg * W  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

