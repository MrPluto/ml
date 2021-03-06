# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_extra_datasets, load_planar_dataset

np.random.seed(1)


# dataset
X, Y = load_planar_dataset()
plt.scatter(X[0,:],X[1,:],c=Y,s=40,cmap=plt.cm.Spectral)

# use logistic regression
# sklearn LR model
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T,Y.T)
LR_predictions = clf.predict(X.T)
# -1 * J(w) / n
accurency = float( (np.dot(Y,LR_predictions)) + np.dot(1 - Y, 1 - LR_predictions)) / float(Y.size) * 100
# about 47%


# methodology to build a neural network
# 1: define neural network structure (input units, hidden units, etc)
# 2: initialize model's parameters
# 3: Loop
#   - Implement forward propagation
#   - Compute Loss
#   - Implement backward propagation
#   - Update parameters (gradient descent)

def layer_sizes(X,Y):
    '''
    the hidden layer has 4 units
    '''
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    return (n_x,n_h,n_y)

def initialize_parameters(n_x,n_h,n_y):
    '''
    '''
    np.random.seed(2)
    W1 = np.random.randn(n_h,n_x) * 0.01 # small weight hight gradient
    b1 = np.zeros(shape=(n_h,1))
    W2 = np.random.randn(n_y,n_h) * 0.01
    b2 = np.zeros(shape=(n_y,1))

    assert( W1.shape == (n_h,n_x))
    #...so on

    parameters = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }
    return parameters

def forward_propagation(X,parameters):

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    Z1 = np.matmul(W1,X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.matmul(W2,A1) + b2
    A2 = sigmoid(Z2)

    cache = {
        'Z1': Z1,
        'A1': A1,
        'Z2': Z2,
        'A2': A2
    }

    return A2, cache

def compute_cost(A2,Y,parameters):

    m = Y.shape[1] # number of example

    W1 = parameters['W1']
    W2 = parameters['W2']

    # np.multiply + np.sum equal to np.dot
    logprobs = np.multiply( Y, np.log(A2) ) + np.multiply( 1 - Y, np.log( 1 - A2) )
    cost = - np.sum(logprobs) / m

    cost = np.squeeze(cost)  # make sure cost is the dimension we expect

    assert(isinstance(cost, float))

    return cost

def backward_propagation(parameters,cache,X,Y):


    W1 = parameters['W1']
    W2 = parameters['W2']

    A1 = cache['A1']
    A2 = cache['A2']

    m = X.shape[1]

    dZ2 = A2 - Y #shape n_y, m
    dW2 = np.dot(dZ2,A1.T) / m
    db2 = np.sum(dZ2,axis=1,keepdims=True) / m

    #dz1 = deravitive of tanh, 1 - a^2
    dZ1 = np.multiply(np.dot(W2.T,(A2 - Y)),1 - np.power(A1,2))
    dW1 = np.dot(dZ1,X.T) / m
    db1 = np.sum(dZ1,axis=1,keepdims=True) / m

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads

def update_parameters(parameters,grads,learning_rate=1.2):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    ### END CODE HERE ###

    # Retrieve each gradient from the dictionary "grads"
    ### START CODE HERE ### (≈ 4 lines of code)
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']


    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def nn_model(X,Y,n_h,num_iterations=10000, print_cost=False):
    np.random.seed(3)

    n_x = layer_sizes(X,Y)[0]
    n_y = layer_sizes(X,Y)[2]

    parameters = initialize_parameters(n_x,n_h,n_y)

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    for i in range(0,num_iterations):
        A2,cache = forward_propagation(X,parameters)
        cost = compute_cost(A2,Y,parameters)

        grads = backward_propagation(parameters,cache,X,Y)
        parameters = update_parameters(parameters,grads)
        if print_cost and i % 1000 ==0:
            print('Cost after iteration %i: %f' % (i,cost))

    return parameters

def predict(parameters,X):
    A2, cache = forward_propagation(X,parameters)
    predictions = np.round(A2)
    return predictions
