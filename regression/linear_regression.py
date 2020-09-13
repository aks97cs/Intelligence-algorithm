from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

x_label = [2, 4, 5, 9]  # feature
y_label = [1.2, 2.8, 5.3, 9]  # target

"""
Hypothesis:
h = theta0 + theta1*x_labRel
"""


def predictive_val(theta0, theta1):
    h_label = []
    for el in range(0, len(x_label)):
        h_label.append(int(theta0 + theta1 * x_label[el]))
    return h_label


def cost_function(theta0, theta1, h_label):
    _cost = 0
    for el in range(len(h_label)):
        _cost += (h_label[el] - y_label[el]) ** 2
    return _cost


def gradient_descent():
    alpha = 0.0000000000000001  # learning rate
    theta0, theta1 = 0, 1  # initial coefficient value
    count = 1
    error = []
    while True:
        print(f"==================epoch = {count} ============")
        print(f"for theta0 = {theta0} and theta1 = {theta1}")
        h_label = predictive_val(theta0, theta1)
        _cost = cost_function(theta0, theta1, h_label)
        print(f"cost = {_cost}")
        print(h_label)
        j_theta0, j_theta1 = 0, 0
        for el in range(0, len(y_label)):
            j_theta0 += -2 * (y_label[el] - h_label[el])
            j_theta1 += -2 * x_label[el] * (y_label[el] - h_label[el])
        theta0, theta1 = (theta0 - (alpha * j_theta0)), (theta1 - (alpha * j_theta1))
        if count == 100000:
            break
        error.append(_cost)
        count += 1
    print("Error: ", min(error))


if __name__ == "__main__":
    gradient_descent()

