import numpy as np
from scipy.optimize import minimize
from scipy.optimize import fmin_cg

from lrCostFunction import lrCostFunction
from lrgradientFunction import lrgradientFunction


def oneVsAll(X, y, num_labels, Lambda):
    """trains multiple logistic regression classifiers and returns all
        the classifiers in a matrix all_theta, where the i-th row of all_theta
        corresponds to the classifier for label i
    """

# Some useful variables
    m, n = X.shape

# You need to return the following variables correctly 
    all_theta = np.zeros((num_labels, n + 1))

# Add ones to the X data matrix
    X = np.column_stack((np.ones((m, 1)), X))

# ====================== YOUR CODE HERE ======================
# Instructions: You should complete the following code to train num_labels
#               logistic regression classifiers with regularization
#               parameter lambda. 
#
# Hint: theta(:) will return a column vector.
#
# Hint: You can use y == c to obtain a vector of 1's and 0's that tell use 
#       whether the ground truth is true/false for this class.
#
# Note: For this assignment, we recommend using fmincg to optimize the cost
#       function. It is okay to use a for-loop (for c = 1:num_labels) to
#       loop over the different classes.

    # Set Initial theta
#    initial_theta = np.zeros((n + 1, 1)) # 401*1
    initial_theta = np.zeros(n + 1) # 401*1
#    print('initial theta shape', initial_theta.shape)

    for k in range(0,num_labels):
#        print('k=', k)
        iclass=k if k else 10 # k=0 correspond iclass=10
        y_temp=np.array([1 if x==iclass else 0 for x in y]).reshape(-1,1)
#        y_temp=np.array([1 if x==k+1 else 0 for x in y]).reshape(-1,1)
        result=fmin_cg(lrCostFunction, fprime=lrgradientFunction, x0=initial_theta, args=(X, y_temp, Lambda), maxiter=50, disp=False, full_output=True)
#        result=minimize(lrCostFunction, initial_theta.reshape(-1), args=(X, y, Lambda),  method='L-BFGS-B', options={"maxiter":20, "disp":False} )
        theta=result[0]
        cost=result[1]
#        print('cost=', cost)
        all_theta[k, :]=theta[:]
    # This function will return theta and the cost

# =========================================================================

    return all_theta

