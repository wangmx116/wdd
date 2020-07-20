from numpy import log
from sigmoid import sigmoid
import numpy as np

def costFunction(theta, X,y):
    """ computes the cost of using theta as the
    parameter for logistic regression and the
    gradient of the cost w.r.t. to the parameters."""

# Initialize some useful values
    m = y.size # number of training examples


# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta.
#               You should set J to the cost.
#               Compute the partial derivatives and set grad to the partial
#               derivatives of the cost w.r.t. each parameter in theta
#
# Note: grad should have the same dimensions as theta

    temp1=np.multiply(y, log(sigmoid(X.dot(theta))))
    temp2=np.multiply(1-y, log(1-sigmoid(X.dot(theta))))
    J=-np.sum(temp1+temp2)/m

#
    return J
