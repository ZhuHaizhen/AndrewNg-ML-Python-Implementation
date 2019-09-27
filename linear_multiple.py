#!/usr/bin/python3
# encoding: utf-8
"""
@author: zhuhz
@file: linear_multiple.py
@time: 2019/9/26 21:13
"""


import numpy as np
import matplotlib.pyplot as plt


def feature_norm(x):
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    x = (x - mu) / sigma

    return x, mu, sigma


data = np.loadtxt('ex1data2.txt', delimiter=',')
print(data.shape)
x0 = data[:, 0:-1]
y0 = data[:, -1][:, np.newaxis]


x0_norm, mu0, sigma0 = feature_norm(x0)
m = y0.shape[0]
x0_mat = np.hstack((np.ones((m, 1)), x0_norm))


def compute_cost_multi(x_mat, y, theta):
    j = 1 / 2 * np.shape(x_mat)[0] * (x_mat.dot(theta) - y).T.dot(x_mat.dot(theta) - y)

    return j


theta0 = np.zeros((x0_mat.shape[1], 1))
j0 = compute_cost_multi(x0_mat, y0, theta0)
print(j0)


def gradient_descent_multi(x_mat, y, alpha, theta, iterations):
    j_history = np.zeros(iterations)
    for i in range(iterations):
        theta = theta - alpha / np.shape(x_mat)[0] * x_mat.T.dot(x_mat.dot(theta) - y)
        j_history[i] = compute_cost_multi(x_mat, y, theta)

    return theta, j_history


alpha0 = 0.15
iterations0 = 50
theta1, j_history1 = gradient_descent_multi(x0_mat, y0, alpha0, theta0, iterations0)

print(theta1)

plt.plot(j_history1, 'b-')
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()

predicted = theta1.T.dot(np.array([[1], [(1650 - mu0[0]) / sigma0[0]], [3 - mu0[1] / sigma0[1]]]))
print(predicted)


def normal_eqn(x_mat, y):
    theta = np.linalg.pinv(x_mat.T.dot(x_mat)).dot(x_mat.T).dot(y)

    return theta


x1_mat = np.hstack((np.ones((m, 1)), x0))
theta2 = normal_eqn(x1_mat, y0)
print(theta2)

predicted0 = theta2.T.dot(np.array([[1], [1650], [3]]))
print(predicted0)
