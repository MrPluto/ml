# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from reg_utils import sigmoid, relu, plot_decision_boundary, initialize_parameters, load_2D_dataset, predict_dec
from reg_utils import compute_cost, predict, backward_propagation, forward_propagation, update_parameters
import sklearn
import sklearn.datasets
import scipy.io
from testCases import *

plt.rcParams['figure.figsize'] = (7.,4.)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

train_X, train_Y, test_X, test_Y = load_2D_dataset()


# Non-regularized model

def model(X,Y,learning_rate=0.3,num_iterations=30000,print_cost=True,lambd=0,keep_prop=1):
    '''
    relu -> relu -> sigmoid
    '''

    grads = {}
    costs = []

    m = X.shape[1]

    layer_dims = [X.shape[0],20,3,1]

    parameters = initialize_parameters(layer_dims)


    for i in range(0,num_iterations):
        if keep_prop == 1:
            a3, cache = forward_propagation(X,parameters)
        elif keep_prop < 1:
            a3, cache = forward_propagation_with_dropout_test_case(X,parameters)
            pass

        if lambd == 0:
            cost = compute_cost(a3,Y)
        else:
            cost = compute_cost_with_regularization(X,parameters,lambd)

        assert(lambd == 0 or keep_prop == 1)

        if lambd == 0 and keep_prop == 1:
            grads = backward_propagation(X,Y,cache)
        elif lambd != 0:
            grads = backward_propagation_with_regularization_test_case(X,Y,lambd)
        elif keep_prop < 1:
            grads = backward_propagation_with_dropout_test_case(X,Y,keep_prop)
            pass

        parameters = update_parameters(parameters,grads,learning_rate)

        if print_cost and i % 1000 == 0:
            print('Cost after iteration {}: {}'.format(i,cost))
            pass
        if print_cost and i % 1000 == 0:
            costs.append(cost)
            pass

        pass

    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations per 1000')
    plt.title('learning_rate =' + str(learning_rate))
    plt.show()

    return parameters


def compute_cost_with_regularization(A3,Y,parameters,lambd):
    '''
    cost function with L2 regularization.
    '''
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]

    cross_entropy_cost= compute_cost(A3,Y)

    L2_regu_cost = lambd * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3))) / (2 * m)

    cost = cross_entropy_cost + L2_regu_cost

    return cost

def backward_propagation_with_regularization(X,Y,cache,lambd):

    pass
