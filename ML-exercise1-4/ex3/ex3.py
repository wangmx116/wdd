## Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all
import scipy.io
import numpy as np
from matplotlib import use
use('TkAgg')
import matplotlib.pyplot as plt

from oneVsAll import oneVsAll
from predictOneVsAll import predictOneVsAll
from displayData import displayData
from predict import predict

np.set_printoptions(threshold=np.inf)

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions 
#  in this exericse:
#
#     lrCostFunction.m (logistic regression cost function)
#     oneVsAll.m
#     predictOneVsAll.m
#     predict.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
print('Exercise 1: Multi-class Classiﬁcation')
## Setup the parameters you will use for this part of the exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
num_labels = 10          # 10 labels, from 1 to 10
                         # (note that we have mapped "0" to label 10)

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
print('Loading and Visualizing Data ...')

data = scipy.io.loadmat('ex3data1.mat') # training data stored in arrays X, y
X = data['X']
y = data['y']
m, _ = X.shape
print("X shape", X.shape)
print("y shape", y.shape)

# Randomly select 100 data points to display
rand_indices = np.random.permutation(range(m))
sel = X[rand_indices[0:100], :]

displayData(sel)

input("Program paused. Press Enter to continue...")

## ============ Part 2: Vectorize Logistic Regression ============
#  In this part of the exercise, you will reuse your logistic regression
#  code from the last exercise. You task here is to make sure that your
#  regularized logistic regression implementation is vectorized. After
#  that, you will implement one-vs-all classification for the handwritten
#  digit dataset.
#

print('Training One-vs-All Logistic Regression...')

Lambda = 0.
all_theta = oneVsAll(X, y, num_labels, Lambda)

input("Program paused. Press Enter to continue...")


## ================ Part 3: Predict for One-Vs-All ================
#  After ...
pred = predictOneVsAll(all_theta, X)
print('pred.shape', pred.shape)

#print('pred', pred)

n_total, n_correct=0., 0.

for irow in range(X.shape[0]):
    n_total+=1
    if pred[irow]==y[irow]:
        n_correct+=1
print('n_total', n_total, 'n_correct', n_correct)
accuracy=100*n_correct/n_total
#accuracy = np.mean(np.double(pred == np.squeeze(y))) * 100
print('\nTraining Set Accuracy: %f\n' % accuracy)

## ================ Part 4: ================
print('\n')
print('Exercise 2: Neural Networks')
print('\n')

## Setup the parameters you will use for this exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10   
                          # (note that we have mapped "0" to label 10)

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
print('Loading and Visualizing Data ...')

data = scipy.io.loadmat('ex3data1.mat')
X = data['X']
y = data['y']
print('X shape', X.shape)
print('y shape', y.shape)
m, _ = X.shape

# Randomly select 100 data points to display
sel = np.random.permutation(range(m))
sel = sel[0:100]

displayData(X[sel,:])

input("Program paused. Press Enter to continue...")

## ================ Part 2: Loading Pameters ================
# In this part of the exercise, we load some pre-initialized 
# neural network parameters.

print('Loading Saved Neural Network Parameters ...')

# Load the weights into variables Theta1 and Theta2
data = scipy.io.loadmat('ex3weights.mat')
Theta1 = data['Theta1']
Theta2 = data['Theta2']

print('Theta1 shape', Theta1.shape)
print('Theta2 shape', Theta2.shape)

## ================= Part 3: Implement Predict =================
#  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.

pred = predict(Theta1, Theta2, X)

n_total, n_correct=0., 0.
for irow in range(X.shape[0]):
    n_total+=1
    if pred[irow]==int(y[irow]):
        n_correct+=1
print('n_total', n_total, 'n_correct', n_correct)
accuracy=100*n_correct/n_total
print('\nTraining Set Accuracy: %f\n' % accuracy)
#print('Training Set Accuracy: %f\n', np.mean(np.double(pred == np.squeeze(y))) * 100) # np.squeeze(y).shape=(5000,)

input("Program paused. Press Enter to continue...")

#  To give you an idea of the network's output, you can also run
#  through the examples one at the a time to see what it is predicting.

#  Randomly permute examples
rp = np.random.permutation(range(m))

plt.figure()
for i in range(0, 5): #range(m):
    # Display
    X2 = X[rp[i],:]
    print('Displaying 5 Example Images')
    X2 = np.matrix(X[rp[i]])
    displayData(X2)

    pred = predict(Theta1, Theta2, X2.getA())
    pred = np.squeeze(pred)
    print('Neural Network Prediction: %d (digit %d)\n' % (pred, np.mod(pred, 10)))
    
    input("Program paused. Press Enter to continue...")
    plt.close()
