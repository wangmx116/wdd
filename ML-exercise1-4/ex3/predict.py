import numpy as np

from sigmoid import sigmoid

def predict(Theta1, Theta2, X):
    """ outputs the predicted label of X given the
    trained weights of a neural network (Theta1, Theta2)
    """

# Useful values
    m, _ = X.shape
    num_labels, _ = Theta2.shape

# ====================== YOUR CODE HERE ======================
# Instructions: Complete the following code to make predictions using
#               your learned neural network. You should set p to a 
#               vector containing labels between 1 to num_labels.
#
# Hint: The max function might come in useful. In particular, the max
#       function can also return the index of the max element, for more
#       information see 'help max'. If your examples are in rows, then, you
#       can use max(A, [], 2) to obtain the max for each row.
#

    X = np.column_stack((np.ones((m, 1)), X))

#    Thetas=[Theta1, Theta2] # len(Theta)=2
#    classes=np.arange(1,11).reshape(-1)
#    p=np.zeros(m)

#    for irow in range(m):
#        temp=X[irow]
#        for i in range(len(Thetas)):
#            Theta=Thetas[i]
#            a=sigmoid(Theta.dot(temp))
#            if i==len(Thetas)-1:
#                p[irow]=classes[np.argmax(a)]
#            a=np.insert(a,0,1)
#            temp=a


    a=sigmoid(X.dot(Theta1.T)) #5000*25
    a=np.column_stack((np.ones((a.shape[0], 1)), a))
    b=sigmoid(a.dot(Theta2.T)) #5000*10
    p=np.argmax(b, axis=1)

# =========================================================================

    return p+1        # add 1 to offset index of maximum in A row

