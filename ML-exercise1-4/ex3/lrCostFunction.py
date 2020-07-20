import sys; 
sys.path.append('C:\\Users\\wangmuxue\\Desktop\\ex2')
from numpy import log
from sigmoid import sigmoid
import numpy as np

def lrCostFunction(theta, X, y, Lambda):
    """computes the cost of using
    theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta.
#               You should set J to the cost.
#
# Hint: The computation of the cost function and gradients can be
#       efficiently vectorized. For example, consider the computation
#
#           sigmoid(X * theta)
#
#       Each row of the resulting matrix will contain the value of the
#       prediction for that example. You can make use of this to vectorize
#       the cost function and gradient computations. 
#

    m = X.shape[0] #5000
    temp1=y.T.dot(log(sigmoid(X.dot(theta.reshape(-1,1))))) # y is 5000*1, X.dot(theta) is 5000*1, temp1 is 1*1
    temp2=(1-y.T).dot(log(1-sigmoid(X.dot(theta.reshape(-1,1)))))

    temp=-(temp1+temp2)/m+theta[1:].T.dot(theta[1:])*Lambda/2/m # 1*1
    J=temp[0]
    # =============================================================

    return J
