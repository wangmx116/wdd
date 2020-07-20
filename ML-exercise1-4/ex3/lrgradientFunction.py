from sigmoid import sigmoid
import numpy as np


def lrgradientFunction(theta, X, y, Lambda):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the gradient of a particular choice of theta.
#               Compute the partial derivatives and set grad to the partial
#               derivatives of the cost w.r.t. each parameter in theta

#    grad=gradientFunction(theta, X, y)+Lambda*theta

    m = X.shape[0] # 5000
    grad=((sigmoid(X.dot(theta.reshape(-1,1)))-y).T.dot(X)).reshape(-1)/m
    grad[1:]=grad[1:]+Lambda*theta[1:]/m #grad.shape=(401,)

# =============================================================

    return grad