# -*- coding: utf-8 -*-

#TLDR:
# SGD ( Stochastic Gradient Descent )  just one training example. ( check mini-batch gradient descent)

# batch GD


import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets

from opt_utils import load_params_and_grads, initialize_parameters, forward_propagation, backward_propagation
from opt_utils import compute_cost, predict, predict_dec, plot_decision_boundary, load_dataset
from testCases import *

%matplotlib inline
plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def update_parameters_with_gd(parameters,grads,learning_rate):
    '''
    '''

    layers = len(parameters) / 2 # one couple of w and b respect one layer

    for layer in range(layers):
        parameters['W' + str( layer + 1 )] = parameters['W' + str( layer + 1 )] - learning_rate * grads['dW' + str( layer + 1 )]
        parameters['b' + str( layer + 1 )] = parameters['b' + str( layer + 1 )] - learning_rate * grads['db' + str( layer + 1 )]
        pass

    return parameters

# Batch Gradient Descent
def gd():
    X = data_input
Y = labels
parameters = initialize_parameters(layers_dims)
for i in range(0, num_iterations):
    # Forward propagation
    a, caches = forward_propagation(X, parameters)
    # Compute cost.
    cost = compute_cost(a, Y)
    # Backward propagation.
    grads = backward_propagation(a, caches, parameters)
    # Update parameters.
    parameters = update_parameters(parameters, grads)
    pass

# Stochastic gradient Descent
def sgd():
    X = data_input
    Y = labels
    parameters = initialize_parameters(layers_dims)
    m = X.shape[1] # number of examples

    for i in range(0, num_iterations):
        for j in range(0,m):
        # Forward propagation
        a, caches = forward_propagation(X[:,j], parameters)
        # Compute cost.
        cost = compute_cost(a, Y[:,j])
        # Backward propagation.
        grads = backward_propagation(a, caches, parameters)
        # Update parameters.
        parameters = update_parameters(parameters, grads
        pass
    pass
