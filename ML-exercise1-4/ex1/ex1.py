from matplotlib import use, cm
use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d
from sklearn import linear_model

from gradientDescent import gradientDescent
from computeCost import computeCost
from warmUpExercise import warmUpExercise
from plotData import plotData
from gradientDescentMulti import gradientDescentMulti
from normalEqn import normalEqn
from featureNormalize import featureNormalize
from show import show

## Machine Learning Online Class - Exercise 1: Linear Regression

#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following modules
#  in this exericse:
#
#     warmUpExercise.py
#     plotData.py
#     gradientDescent.py
#     computeCost.py
#     gradientDescentMulti.py
#     computeCostMulti.py
#     featureNormalize.py
#     normalEqn.py
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
# x refers to the population size in 10,000s
# y refers to the profit in $10,000s

print('Exercise 1: Linear Regression')

# ==================== Part 1: Basic Function ====================
# Complete warmUpExercise.py
print('Running warmUpExercise ...')
print('5x5 Identity Matrix:')
warmup = warmUpExercise()
print(warmup)
input("Program paused. Press Enter to continue...")

# ======================= Part 2: Plotting =======================
data = np.loadtxt('ex1data1.txt', delimiter=',')
m = data.shape[0]
X = np.vstack(zip(np.ones(m),data[:,0]))
y = data[:, 1]

# Plot Data
# Note: You have to complete the code in plotData.py
print('Plotting Data ...')
plotData(data)
show()

input("Program paused. Press Enter to continue...")

# =================== Part 3: Gradient descent ===================
print('Running Gradient Descent ...')
theta = np.zeros(2)

# compute and display initial cost
J = computeCost(X, y, theta)
print('cost: %0.4f ' % J)

# Some gradient descent settings
iterations = 1500
alpha = 0.01

# run gradient descent
theta, J_history = gradientDescent(X, y, theta, alpha, iterations)

# print theta to screen
print('Theta found by gradient descent: ')
print('%s %s \n' % (theta[0], theta[1]))

# Plot the linear fit
#plt.figure()
plotData(data)
plt.plot(X[:, 1], X.dot(theta), '-', label='Linear regression')
plt.legend(loc='upper right', shadow=True, fontsize='x-large', numpoints=1)
show()

input("Program paused. Press Enter to continue...")

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.array([1, 3.5]).dot(theta)
predict2 = np.array([1, 7]).dot(theta)
print('For population = 35,000, we predict a profit of {:.4f}'.format(predict1*10000))
print('For population = 70,000, we predict a profit of {:.4f}'.format(predict2*10000))

# ============= Part 4: Visualizing J(theta_0, theta_1) =============
print('Visualizing J(theta_0, theta_1) ...')

# Grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, X.shape[0])
theta1_vals = np.linspace(-1, 4, X.shape[0])

# initialize J_vals to a matrix of 0's
J_vals=np.array(np.zeros(X.shape[0]).T)

for i in range(theta0_vals.size):
    col = []
    for j in range(theta1_vals.size):
        t = np.array([theta0_vals[i],theta1_vals[j]])
        col.append(computeCost(X, y, t.T))
    J_vals=np.column_stack((J_vals,col))

# Because of the way meshgrids work in the surf command, we need to
# transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals[:,1:].T
theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)

# Surface plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(theta0_vals, theta1_vals, J_vals, rstride=8, cstride=8, alpha=0.3,
                cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_xlabel(r'$\theta_0$')
ax.set_ylabel(r'$\theta_1$')
ax.set_zlabel(r'J($\theta$)')
show()

input("Program paused. Press Enter to continue...")

# Contour plot
#plt.figure()

# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
ax = plt.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 20))
plt.clabel(ax, inline=1, fontsize=10)
plt.xlabel(r'$\theta_0$')
plt.ylabel(r'$\theta_1$')
plt.plot(0.0, 0.0, 'rx', linewidth=2, markersize=10)
show()

input("Program paused. Press Enter to continue...")

# =============Use Scikit-learn =============
regr = linear_model.LinearRegression(fit_intercept=False, normalize=True)
regr.fit(X, y)

print('Theta found by scikit: ')
print('%s %s \n' % (regr.coef_[0], regr.coef_[1]))

predict1 = np.array([1, 3.5]).dot(regr.coef_)
predict2 = np.array([1, 7]).dot(regr.coef_)
print('For population = 35,000, we predict a profit of {:.4f}'.format(predict1*10000))
print('For population = 70,000, we predict a profit of {:.4f}'.format(predict2*10000))

#plt.figure()
plotData(data)
plt.plot(X[:, 1],  X.dot(regr.coef_), '-', color='black', label='Linear regression wit scikit')
plt.legend(loc='upper right', shadow=True, fontsize='x-large', numpoints=1)
show()

input("Program paused. Press Enter to continue...")

# ================ Part 1: Feature Normalization ================
print('\n')
print('Exercise 2: Linear regression with multiple variables')
print('\n')
print('Loading data ...')

# Load Data
data = np.loadtxt('ex1data2.txt', delimiter=',')
X = data[:, :2]
y = data[:, 2]
m = y.T.size


# Print out some data points
print('First 10 examples from the dataset:')
print(np.column_stack( (X[:10], y[:10]) ))
input("Program paused. Press Enter to continue...")

# Scale features and set them to zero mean
print('Normalizing Features ...')

X, mu, sigma = featureNormalize(X)
print('[mu] [sigma]')
print(mu, sigma)

# Add intercept term to X
X = np.concatenate((np.ones((m, 1)), X), axis=1)


# ================ Part 2: Gradient Descent ================
#
# ====================== YOUR CODE HERE ======================
# Instructions: We have provided you with the following starter
#               code that runs gradient descent with a particular
#               learning rate (alpha).
#
#               Your task is to first make sure that your functions -
#               computeCost and gradientDescent already work with
#               this starter code and support multiple variables.
#
#               After that, try running gradient descent with
#               different values of alpha and see which one gives
#               you the best result.
#
#               Finally, you should complete the code at the end
#               to predict the price of a 1650 sq-ft, 3 br house.
#
# Hint: By using the 'hold on' command, you can plot multiple
#       graphs on the same figure.
#
# Hint: At prediction, make sure you do the same feature normalization.
#

print('Running gradient descent ...')

# Choose some alpha value
alpha = 0.01
num_iters = 400

# Init Theta and Run Gradient Descent 
theta = np.zeros(3)
theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)

# Plot the convergence graph
plt.plot(J_history, '-b')
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
show()
input("Program paused. Press Enter to continue...")

# Display gradient descent's result
print('Theta computed from gradient descent: ')
print(theta)

# Estimate the price of a 1650 sq-ft, 3 br house
price = np.array([1,3,1650]).dot(theta)

print('Predicted price of a 1650 sq-ft, 3 br house')
print('(using gradient descent): ')
print(price)

input("Program paused. Press Enter to continue...")

# ================ Part 3: Normal Equations ================

# ====================== YOUR CODE HERE ======================
# Instructions: The following code computes the closed form
#               solution for linear regression using the normal
#               equations. You should complete the code in
#               normalEqn.m
#
#               After doing so, you should complete this code
#               to predict the price of a 1650 sq-ft, 3 br house.
#

print('Solving with normal equations...')

# Load Data
data = np.loadtxt('ex1data2.txt', delimiter=',')
X = data[:, :2]
y = data[:, 2]
m = y.T.size

# Add intercept term to X
X = np.concatenate((np.ones((m,1)), X), axis=1)

# Calculate the parameters from the normal equation
theta = normalEqn(X, y)

# Display normal equation's result
print('Theta computed from the normal equations:')
print(' %s \n' % theta)

# Estimate the price of a 1650 sq-ft, 3 br house
price = np.array([1, 3, 1650]).dot(theta)

# ============================================================

print("Predicted price of a 1650 sq-ft, 3 br house ")
print('(using normal equations):\n $%f\n' % price)

input("Program paused. Press Enter to continue...")
