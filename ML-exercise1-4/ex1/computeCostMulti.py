import numpy as np

def computeCostMulti(X, y, theta):
    """
     Compute cost for linear regression with multiple variables
       J = computeCost(X, y, theta) computes the cost of using theta as the
       parameter for linear regression to fit the data points in X and y
    """
    m = y.size
    J = 0
# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta
#               You should set J to the cost.


# =========================================================================

    temp=np.power((X.dot(theta)-y).reshape(-1,1), 2)
    J=np.sum(temp)/2/m
    
    return J
