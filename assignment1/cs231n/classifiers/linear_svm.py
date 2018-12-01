import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero
  dW2 = np.zeros(W.shape)

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:, y[i]] += -X[i, :].T    
        dW[:, j] += X[i, :].T
#         print(dW2[:, y[i]])
#         print(dW2[:, j])
#     print(dW2)

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
#   for i in range(num_train):
#     dtest = np.zeros((W.shape))
#     for j in range(num_classes):
#         if j == y[i]:
#             dW[:,j] -= (np.sum((X[i].dot(W) - X[i].dot(W[:,y[i]]) + 1) > 0) - 1) * X[i]
# #             print(dW[:,j])            
#             continue
# #         dW[:,y[i]] -= ((W[:,j].T.dot(X[i]) - W[:,y[i]].T.dot(X[i]) + 1) > 0) * X[i]    
# #         dtest[:,y[i]] -= ((W[:,j].T.dot(X[i]) - W[:,y[i]].T.dot(X[i]) + 1) > 0) * X[i]
# #         print(((W[:,j].T.dot(X[i]) - W[:,y[i]].T.dot(X[i]) + 1) > 0))
#         dW[:,j] += ((W[:,j].T.dot(X[i]) - W[:,y[i]].T.dot(X[i]) + 1) > 0) * X[i]        
# #         dW[:,j] += ((X[i].dot(W[:,j]) - X[i].dot(W[:,y[i]]) + 1) > 0) * X[i, :].T
# #         print((W[:,j].T.dot(X[i]) - W[:,y[i]].T.dot(X[i]) + 1) > 0)
# #         print("@",dW[:,j])
# #         print(X[i])
# #     print("@",np.sum((X[i].dot(W) - X[i].dot(W[:,y[i]]) + 1) > 0))
# #     print("--",dtest[:,y[i]])
# #     dW += dW
# #     print(dW)
#   dW /= num_train
# #   print(dW)
# #   print(dW2)
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  delta = 1

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  N = X.shape[0]
  C = W.shape[1]
  
  Z = X.dot(W)
  Sy = Z[list(range(N)), y]
  Diff = (Z - Sy.reshape(Sy.shape[0],-1) + delta)
  Mask = (Diff > 0)
  Sum_h = np.sum(np.multiply(Mask, Diff), axis = 1)
  Sum_h -= 1
  loss = np.sum(Sum_h, axis = 0)
  loss /= N
  loss += reg * np.sum(W * W)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
#   Mask_num = np.ones((Mask.shape))
#   Mask_num *= Mask
#   Mask_num[np.arange(N), y] = -(np.sum(Mask, axis = 1) - 1)  # note that dimesion
#   dW = X.T.dot(Mask_num)
#   dW /= N
#   dW += 2 * reg * W

  Mask_num = Mask.astype(int)
  Mask_num[np.arange(N), y] = -(np.sum(Mask_num, axis = 1) - 1)  # note that dimesion
  dW = X.T.dot(Mask_num)
  dW /= N
  dW += 2 * reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
