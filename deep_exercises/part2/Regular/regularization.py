# -*- coding: utf-8 -*-

# TLDR
# L2:
#
#   在 loss function 中加入一项 回归项  (regularization term)
#   在 back prop 对应的 W 中加入额外的梯度
#   权重变得更小  (weight decay)

# Dropout: 随机的关闭一些 neuron units. (初衷: 使其模型对某一节点的依赖减小 -- cant rely on any one feature)
#   只在测试中使用，(不要在测试时使用)
#   forward and backward propagation 中都应该使用 (缓存在 forward 中计算出来的 dropout 节点)
#   记得除掉 keep_prob 以保证 输出值与不加dropout时相同
#
# Conclusion: 可以减少 overfitting, 会使 weight 变得更小

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


# Non-regularized model if lambd = 0 and keep_prob = 1

def model(X,Y,learning_rate=0.3,num_iterations=30000,print_cost=True,lambd=0,keep_prob=1):
    '''
    relu -> relu -> sigmoid
    '''

    grads = {}
    costs = []

    m = X.shape[1]

    layer_dims = [X.shape[0],20,3,1]

    parameters = initialize_parameters(layer_dims)


    for i in range(0,num_iterations):
        if keep_prob == 1:
            a3, cache = forward_propagation(X,parameters)
        elif keep_prob < 1:
            a3, cache = forward_propagation_with_dropout(X,parameters,keep_prob)
            pass

        if lambd == 0:
            cost = compute_cost(a3,Y)
        else:
            cost = compute_cost_with_regularization(a3,Y,parameters,lambd)

        assert(lambd == 0 or keep_prob == 1)

        if lambd == 0 and keep_prob == 1:
            grads = backward_propagation(X,Y,cache)
        elif lambd != 0:
            grads = backward_propagation_with_regularization(X,Y,cache,lambd)
        elif keep_prob < 1:
            grads = backward_propagation_with_dropout(X,Y,cache,keep_prob)
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

    return parameters, costs


def compute_cost_with_regularization(A3,Y,parameters,lambd):
    '''
    cost function with L2 regularization.
    L2: ||W||^2 * 2 / m
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

    # dw(regularization term) = d(1/2 * r/m * W^2) = r/m * W

    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y

    dW3 = 1. / m * np.dot(dZ3,A2.T) + (lambd * W3) / m

    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))   # d(relu)/dx = 1 when x>=0 and 0 when x<0
    ### START CODE HERE ### (approx. 1 line)
    dW2 = 1. / m * np.dot(dZ2, A1.T) + (lambd * W2) / m
    ### END CODE HERE ###
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    ### START CODE HERE ### (approx. 1 line)
    dW1 = 1. / m * np.dot(dZ1, X.T) + (lambd * W1) / m
    ### END CODE HERE ###
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients


def forward_propagation_with_dropout(X,parameters,keep_prob):
    '''
    step 1: create a random matrix d with shape of a
    step 2: if value < keep_prob then False, otherwise True (mark the neuron active whether or not)
    step 3: set a = a * d. (disable some neuron)
    step 4: set a = a / keep_prob  (keep the loss the same with non-dropout, also called Inverted dropout)
    '''

    np.random.seed(1)

    # retrieve parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)

    D1 = np.random.randn(A1.shape[0],A1.shape[1])              # Step 1: initialize matrix D2 = np.random.rand(..., ...)
    D1 = D1 > keep_prob                         # Step 2: convert entries of D2 to 0 or 1 (using keep_prob as the threshold)
    A1 = A1 * D1                                # Step 3: shut down some neurons of A2
    A1 = A1 / keep_prob                         # Step 4: scale the value of neurons that haven't been shut down

    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    ### START CODE HERE ### (approx. 4 lines)
    D2 = np.random.rand(A2.shape[0], A2.shape[1])
    D2 = D2 < keep_prob
    A2 = A2 * D2
    A2 = A2 / keep_prob
    ### END CODE HERE ###
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    cache = (Z1,  A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3, D1, D2)

    return A3, cache

def backward_propagation_with_dropout(X,Y,cache,keep_prob):

    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3, D1, D2) = cache

    dZ3 = A3 - Y
    dW3 = 1 / m * np.dot(dZ3,A2.T)
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    ### START CODE HERE ### (≈ 2 lines of code)
    dA2 = dA2 * D2              # Step 1: Apply mask D2 to shut down the same neurons as during the forward propagation
    dA2 = dA2 / keep_prob       # Step 2: Scale the value of neurons that haven't been shut down

    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1. / m * np.dot(dZ2, A1.T)
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)

    dA1 = dA1 * D1
    dA1 = dA1 / keep_prob

    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1. / m * np.dot(dZ1, X.T)
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients
