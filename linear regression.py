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


data = pd.read_csv('ex1data1.txt', sep=',')
x0 = np.array(data.iloc[:, 0])
y0 = np.array(data.iloc[:, 1])
plot_data(x0, y0)

m = np.shape(y0)[0]
x0_mat = pd.concat([np.ones(m), x0], axis=1, ignore_index=True)


def compute_cost(x_mat, y, theta):
    j = 1 / (2 * np.shape(x_mat)[0]) * np.sum((theta.T.dot(x_mat) - y) ** 2)
    return j


theta0 = np.zeros((2, 1))
compute_cost(x0_mat, y0, theta0)


def grad_descent(x_mat, y, theta, alpha, iterations):
    j_history = np.zeros(iterations)
    for i in range(iterations):
        theta = theta - alpha / np.shape(x_mat)[0] * x_mat.T.dot(theta.T.dot(x_mat) - y)
        j_history[i] = compute_cost(x_mat, y, theta)

    return theta, j_history


iter0 = 1500
alpha0 = 0.01
theta1, j_history1 = grad_descent(x0_mat, y0, theta0, alpha0, iter0)

plt.subplot(111)
plt.scatter(x0, y0, 'rx')
plt.plot(x0, theta1.T.dot(x0), 'b-')
plt.legend(loc=4)
plt.show()
