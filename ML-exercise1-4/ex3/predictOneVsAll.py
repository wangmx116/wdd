import numpy as np

from sigmoid import sigmoid

def predictOneVsAll(all_theta, X):
    """will return a vector of predictions
  for each example in the matrix X. Note that X contains the examples in
  rows. all_theta is a matrix where the i-th row is a trained logistic
  regression theta vector for the i-th class. You should set p to a vector
  of values from 1..K (e.g., p = [1 3 1 2] predicts classes 1, 3, 1, 2
  for 4 examples) """

    m = X.shape[0]

    # You need to return the following variables correctly
#    p = np.zeros((m, 1))

    # Add ones to the X data matrix
    X = np.column_stack((np.ones((m, 1)), X))

# ====================== YOUR CODE HERE ======================
# Instructions: Complete the following code to make predictions using
#               your learned logistic regression parameters (one-vs-all).
#               You should set p to a vector of predictions (from 1 to
#               num_labels).
#
# Hint: This code can be done all vectorized using the max function.
#       In particular, the max function can also return the index of the 
#       max element, for more information see 'help max'. If your examples 
#       are in rows, then, you can use max(A, [], 2) to obtain the max 
#       for each row.
#       

    temp=sigmoid(X.dot(all_theta.T)) #5000*10
#    print(sigmoid(X.dot(all_theta.T))[1,:])
    p=np.argmax(temp, axis=1)

#    classes=np.arrange(1, 11)
#    p=np.zeros(m).reshape(-1)
#    for irow in range(m):
#            a=sigmoid(all_theta.dot(X[irow]))
#            p[irow]=classes[np.argmax(a)]

# =========================================================================

    return np.array([x if x else 10 for x in p])# + 1    # add 1 to offset index of maximum in A row
