# Logistic Regression
from matplotlib import use

use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import fmin
from scipy.optimize import fmin_tnc

import pandas as pd

from costFunction import costFunction
from gradientFunction import gradientFunction
from sigmoid import sigmoid
from predict import predict
from ml import mapFeature, plotData, plotDecisionBoundary
from costFunctionReg import costFunctionReg
from gradientFunctionReg import gradientFunctionReg
from show import show

## Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the second part
#  of the exercise which covers regularization with logistic regression.
#
#  You will need to complete the following functions in this exericse:
#
#     sigmoid.py
#     costFunction.py
#     gradientFunction.py
#     predict.py
#     costFunctionReg.py
#     gradientFunctionReg.py
#     n.b. This files differ in number from the Octave version of ex2.
#          This is due to the scipy optimization taking only scalar
#          functions where fmiunc in Octave takes functions returning
#          multiple values.
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

from ml import plotData, plotDecisionBoundary
# Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label.
print('Exercise 1: Logistic Regression')

data = np.loadtxt('ex2data1.txt', delimiter=',')
X = data[:, 0:2]
y = data[:, 2]

# ==================== Part 1: Plotting ====================

print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.')

plotData(X, y)
plt.legend(['Admitted', 'Not admitted'], loc='upper right', shadow=True, fontsize='x-large', numpoints=1)

plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
show()
input("Program paused. Press Enter to continue...")


# # ============ Part 2: Compute Cost and Gradient ============
# #  Setup the data matrix appropriately, and add ones for the intercept term
m, n = X.shape

# Add intercept term to x and X_test
X = np.concatenate((np.ones((m, 1)), X), axis=1)

# Initialize fitting parameters
initial_theta = np.zeros(n + 1)

# Compute and display initial cost and gradient
cost = costFunction(initial_theta, X, y)
print('Cost at initial theta (zeros): %f' % cost)

grad = gradientFunction(initial_theta, X, y)
print('Gradient at initial theta (zeros): ' + str(grad))

input("Program paused. Press Enter to continue...")

# ============= Part 3: Optimizing using scipy  =============
#res = minimize(costFunction, initial_theta, method='TNC',
#               jac=False, args=(X, y), options={'gtol': 1e-3, 'disp': True, 'maxiter': 1000})

#theta = res.x
#cost = res.fun

res=fmin(costFunction, initial_theta, (X, y), maxiter=400, full_output=True)
theta=res[0]
cost=res[1]

# Print theta to screen
print('Cost at theta found by scipy: %f' % cost)
#print('theta:', ["%0.4f" % i for i in theta])

# Plot Boundary
plotDecisionBoundary(theta, X, y)

# Labels and Legend
plt.legend(['Admitted', 'Not admitted'], loc='upper right', shadow=True, fontsize='x-large', numpoints=1)
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
show()

input("Program paused. Press Enter to continue...")

#  ============== Part 4: Predict and Accuracies ==============

#  Predict probability for a student with score 45 on exam 1
#  and score 85 on exam 2

prob = sigmoid(np.array([1, 45, 85]).dot(theta))
print('For a student with scores 45 and 85, we predict an admission probability of %f' % prob)

# Compute accuracy on our training set
p = predict(theta, X)
acc = 1.0*np.where(p == y)[0].size/len(p) * 100
print('Train Accuracy: %f' % acc)

input("Program paused. Press Enter to continue...")

#  ============== Part 5: Regularized Logistic Regression ==============
print('\n')
print('Exercise 2: Regularized Logistic Regression')
print('\n')
def optimize(Lambda):

#    result = minimize(costFunctionReg, initial_theta, method='L-BFGS-B',
#               jac=gradientFunctionReg, args=(X, y, Lambda), #X.as_matrix()
#               options={'gtol': 1e-4, 'disp': False, 'maxiter': 1000})

#    result=fmin(costFunctionReg, initial_theta, (X, y, Lambda), maxiter=500, full_output=True)

    result=minimize(costFunctionReg, initial_theta, args=(X, y, Lambda),  method='L-BFGS-B', options={"maxiter":500, "disp":False} )

    return result


# Plot Boundary
def plotBoundary(theta, X, y):

    plotDecisionBoundary(theta, X, y)
    plt.title(r'$\lambda$ = ' + str(Lambda))

    # Labels and Legend
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    show()

# Initialization

# Load Data
#  The first two columns contains the X values and the third column
#  contains the label (y).

data = pd.read_csv('ex2data2.txt', header=None, names=[1,2,3])
X = data[[1, 2]]
y = data[[3]]
X=X.values
y=y.values
y=np.array(y).reshape(-1)

plotData(X, y)

# Labels and Legend
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
show()
input("Program paused. Press Enter to continue...")


# =========== Part 1: Regularized Logistic Regression ============

# Add Polynomial Features

# Note that mapFeature also adds a column of ones for us, so the intercept
# term is handled
X=pd.DataFrame(X)
X = X.apply(mapFeature, axis=1)
print(X.shape)
X=X.values

# Initialize fitting parameters
initial_theta = np.zeros(X.shape[1])

# Set regularization parameter lambda to 1
Lambda = 0.0

# Compute and display initial cost and gradient for regularized logistic
# regression
cost = costFunctionReg(initial_theta, X, y, Lambda)

print('Cost at initial theta (zeros): %f' % cost)

# ============= Part 2: Regularization and Accuracies =============

# Optimize and plot boundary

Lambda = 1.0
result = optimize(Lambda)
theta = result.x
cost = result.fun

#theta = result[0]
#cost = result[1]

# Print to screen
print('lambda = ' + str(Lambda))
print('Cost at theta found by scipy: %f' % cost)
#print('theta:', ["%0.4f" % i for i in theta])

input("Program paused. Press Enter to continue...")

plotBoundary(theta, X, y)

# Compute accuracy on our training set
p = np.round(sigmoid(X.dot(theta)))
acc = np.mean(np.where(p == y.T,1,0)) * 100
print('Train Accuracy: %f' % acc)

input("Program paused. Press Enter to continue...")

# ============= Part 3: Optional Exercises =============


for Lambda in np.arange(0.0,100.1,100.0):
    result = optimize(Lambda)
    theta = result.x
#    theta = result[0]
    print('lambda = ' + str(Lambda))
#    print('theta:', ["%0.4f" % i for i in theta])
    plotBoundary(theta, X, y)
input("Program paused. Press Enter to continue...")