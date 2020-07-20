import numpy as np

from numpy import log
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient


def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda):

    """computes the cost and gradient of the neural network. The
  parameters for the neural network are "unrolled" into the vector
  nn_params and need to be converted back into the weight matrices.

  The returned parameter grad should be a "unrolled" vector of the
  partial derivatives of the neural network.
    """

# Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
# for our 2 layer neural network
# Obtain Theta1 and Theta2 back from nn_params
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                       (hidden_layer_size, input_layer_size + 1), order='F').copy() # (25, 401)

    Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                       (num_labels, (hidden_layer_size + 1)), order='F').copy() # (10, 26)



# Setup some useful variables
    m, _ = X.shape


# ====================== YOUR CODE HERE ======================
# Instructions: You should complete the code by working through the
#               following parts.
#
# Part 1: Feedforward the neural network and return the cost in the
#         variable J. After implementing Part 1, you can verify that your
#         cost function computation is correct by verifying the cost
#         computed in ex4.m
#
# Part 2: Implement the backpropagation algorithm to compute the gradients
#         Theta1_grad and Theta2_grad. You should return the partial derivatives of
#         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
#         Theta2_grad, respectively. After implementing Part 2, you can check
#         that your implementation is correct by running checkNNGradients
#
#         Note: The vector y passed into the function is a vector of labels
#               containing values from 1..K. You need to map this vector into a 
#               binary vector of 1's and 0's to be used with the neural network
#               cost function.
#
#         Hint: We recommend implementing backpropagation using a for-loop
#               over the training examples if you are implementing it for the 
#               first time.
#
# Part 3: Implement regularization with the cost function and gradients.
#
#         Hint: You can implement this around the code for
#               backpropagation. That is, you can compute the gradients for
#               the regularization separately and then add them to Theta1_grad
#               and Theta2_grad from Part 2.
#

    X=np.column_stack((np.ones((m, 1)), X)) #(5000, 401) a1=X
    a2=sigmoid(X.dot(Theta1.T)) #(5000, 25)
    a2=np.column_stack((np.ones((a2.shape[0], 1)), a2)) #(5000, 26)
    a3=sigmoid(a2.dot(Theta2.T)) #(5000, 10) output a3
    
    J=0.
    delta3=np.zeros((m, num_labels))
    for k in range(num_labels):
#        iclass=k if k else 10
#        y_temp=np.array([1 if x==iclass else 0 for x in y])#.reshape(1,-1) # y_temp (5000,)
        y_temp=np.array([1 if x==k+1 else 0 for x in y]).reshape(1,-1)
        J=J-y_temp.dot(log(a3[:,k]))-(1-y_temp).dot(log(1.-a3[:,k])) #(1,)
        delta3[:,k]=a3[:,k]-y_temp # (5000, 10)
#        Theta2_grad
    reg=(np.sum(Theta1[:,1:]**2.)+np.sum(Theta2[:, 1:]**2.))*Lambda/2./m #(1,)

    J=J/m+reg
    # -------------------------------------------------------------

    # =========================================================================
    Theta1_grad=np.zeros((hidden_layer_size, input_layer_size+1)) # (25, 401)
    Theta2_grad=np.zeros((num_labels, hidden_layer_size+1)) # (10, 26)

    delta2=(delta3.dot(Theta2[:, 1:]))*sigmoidGradient(X.dot(Theta1.T)) # (5000, 25)
    Theta1_grad=Theta1_grad+delta2.T.dot(X) # (25, 401)
    Theta2_grad=Theta2_grad+delta3.T.dot(a2) # (10, 26)

    # Unroll gradient
    grad = np.hstack((Theta1_grad.T.ravel(), Theta2_grad.T.ravel())) 


    return J, grad