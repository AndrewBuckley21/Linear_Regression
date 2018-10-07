# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style; style.use('ggplot')

# load data
data = pd.read_csv('~/NDL/linear_regression_data.csv')

# set up data
X = data.X
y = data.y
m = y.size

# visualize initial scatter plot
# plt.scatter(X, y, color='blue')
# plt.show()

# account for y-intercept
X = np.hstack(((np.ones((m, 1)),
                X.values.reshape(m, 1))))
y = y.values.reshape(m, 1)

# initialize parameters
theta = np.zeros((2, 1))

def predict():
    return np.matmul(X, theta)

def cost():
    return (1/(2*m)) * np.sum((predict()-y) ** 2)

def gradient():
    return (1/m) * np.matmul(X.T, (predict() - y))

# specific gradient descent parameters
num_iterations = 1500
learning_rate = 0.01
# perform gradient descent
for i in range(num_iterations):
    theta -= learning_rate * gradient()

    X_pt = [min(X[:, 1]), max(X[:, 1])]
    y_pt = [min(predict()), max(predict())]

    plt.scatter(X[:, 1], y, color='blue')
    plt.plot(X_pt, y_pt)
    plt.show()