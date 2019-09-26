#!/usr/bin/python3
# encoding: utf-8
"""
@author: zhuhz
@file: linear regression.py
@time: 2019/9/23 21:04
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_data(x, y):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x, y, color='r', marker='x')
    ax.set_xlabel('Profit in $ 10,000s')
    ax.set_ylabel('Population of City in 10,000s')
    plt.show()


data = pd.read_csv('ex1data1.txt', sep=',', header=None)
data = np.array(data)
print(data.shape)

x0 = data[:, 0][:, np.newaxis]
y0 = data[:, 1][:, np.newaxis]
plot_data(x0, y0)

m = np.shape(y0)[0]
x0_mat = np.hstack((np.ones((m, 1)), x0))


def compute_cost(x_mat, y, theta):
    j = 1 / (2 * np.shape(x_mat)[0]) * np.sum((x_mat.dot(theta) - y) ** 2)
    return j


theta0 = np.zeros((2, 1))
print(compute_cost(x0_mat, y0, theta0))
print(compute_cost(x0_mat, y0, np.array([[-1], [2]])))


def grad_descent(x_mat, y, theta, alpha, iterations):
    j_history = np.zeros(iterations)
    for i in range(iterations):
        theta = theta - alpha / np.shape(x_mat)[0] * x_mat.T.dot(x_mat.dot(theta) - y)
        j_history[i] = compute_cost(x_mat, y, theta)

    return theta, j_history


iter0 = 1500
alpha0 = 0.01
theta1, j_history1 = grad_descent(x0_mat, y0, theta0, alpha0, iter0)
print(theta1)
print(j_history1)

plt.subplot(111)
plt.scatter(x0, y0, color='r', marker='x')
plt.plot(x0, x0_mat.dot(theta1), 'b-')
plt.legend(loc=4)
plt.xlabel('Profit in $ 10,000s')
plt.ylabel('Population of City in 10,000s')
plt.show()

theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
j_vals = np.zeros((100, 100))

for i in range(100):
    for j in range(100):
        t = np.array([[theta0_vals[i]], [theta1_vals[j]]])
        j_vals[i, j] = compute_cost(x0_mat, y0, t)

plt.subplot(111)
plt.contour(theta0_vals, theta1_vals, j_vals.T, np.logspace(-2, 3, 20))
plt.xlabel(r'$\theta_0$')
plt.ylabel(r'$\theta_1$')
plt.plot(theta1[0, 0], theta1[1, 0], color='r', marker='x')
plt.show()
