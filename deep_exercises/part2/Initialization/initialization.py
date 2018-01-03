# -*- coding: utf-8 -*-

# A well chosen initialization can:
# 1: speed up the convergence of gradient descent
# 2: increase the odds of gradient descent converging to lowe training (and generalization) error


import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from init_utils import sigmoid, relu, compute_loss, forward_propagation,backward_propagation
from init_utils import update_parameters, predict, load_dataset, plot_decision_boundary, predict_dec

plt.rcParams['figure.figsize'] = (7.0, 4.0) #default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

train_X, train_Y, test_X, test_Y = load_dataset()

def model(X,Y,learning_rate=0.01, num_iterations=15000,print_cost=True,initialization='he'):
    '''
    A three layers neural network: linear -> relu -> linear -> relu -> linear -> sigmoid.

    return: parameters learnt by this model
    '''

    grads = {}
    costs = []
    m = X.shape[1]
    layers_dims = [x.shape[0],10,5,1]

    if initialization == 'zeros':
        parameters = initialize_paramters_zeros(layers_dims)
    elif initialization == 'random':
        parameters = initialize_paramters_random(layers_dims)
    elif initialization == 'he':
        parameters = initialize_paramters_he(layers_dims)
        pass

    for i in range(0,num_iterations):
        a3, cache = forward_propagation(X,parameters)

        cost = compute_loss(a3,Y)

        grads = backward_propagation(X,Y,cache)

        parameters = update_parameters(parameters,grads,learning_rate)

        if print_cost && i % 1000 == 0:
            costs.append(const)
            print("Cost after iteration {}: {}".format(i, cost))
            pass
        pass

    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title('Learning rate =' + str(learning_rate))
    plt.show()

    return parameters

def initialize_paramters_zeros(layers_dims):
    parameters = {}
    lenth = len(layers_dims)

    for l in range(1,length):
        parameters['W' + str(l)] = np.zeros(layers_dims[l],layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros(layers_dims[l],1)
    pass

def initialize_paramters_random(layers_dims):
    np.random.seed(3)
    parameters = {}
    lenth = len(layers_dims)

    for l in range(1,length):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1]) * 10
        parameters['b' + str(l)] = np.random.randn(layers_dims[l],1)
    pass


def initialize_paramters_he(layers_dims):
    np.random.seed(3)
    parameters = {}
    lenth = len(layers_dims)

    for l in range(1,length):
        # https://www.leiphone.com/news/201703/3qMp45aQtbxTdzmK.html
        heScalar = np.sqrt(2 / layers_dims[l - 1]) # similar to xavier initialization which is np.sqrt(1 / layers_dims[l - 1])

        parameters['W' + str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1]) * 10
        parameters['b' + str(l)] = np.random.randn(layers_dims[l],1)
    pass
    pass
