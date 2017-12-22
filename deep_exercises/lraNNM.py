# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy
import h5py
from scipy import ndimage
from PIL import Image
from lr_utils import load_dataset

#step 1 , prepare data
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

#show Image
index = 25
plt.figure()
plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y[:,index]) + ", it's a '" + classes[np.squeeze(train_set_y[:,index])].decode("utf-8") +  "' picture.")

#get the shape and reshape the data: x.shape => (a,b,c,d)
#(num,px,px,channels)

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

#reshape , an image is a vector (64,64,3) => (64*64*3,1)
train_set_x_flattern = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T # equal to reshape(a,b*c*d).T
test_set_x_flattern = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
#should be shape (b*c*d,a)

#standardize dataset, for image, just divide 255 cause this is the maximum value for color
train_set_x = train_set_x_flattern / 255.
test_set_x = test_set_x_flattern / 255.


#step 2 , make algorithm
# we use sigmod as our activation function and cross entropy as our loss function

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

def initialize_with_zeros(dim):
    '''
    This create a vector of zeros of shape(dim,1) for w and initializes b to 0
    Notice: maybe we should not initial zero cause it may cause semmetry problem
            we can initializes with a random small value instead of zeros
            small value for better gradient descent
    '''
    w = np.zeros(shape=(dim,1)) #np.random.rand(dim,1) * 0.01
    b = 0

    assert(w.shape == (dim,1))
    assert(isinstance(b,float) or isinstance(b,int))

    return w,b

#forward and backward propagation

def propagate(w,b,X,Y):
    '''
    Implement the cost function and gradient

    w: weight, a numpy array with shape (num_px * num_px * 3, 1)
    b: bias
    X: train set. shape (num_px * num_px * 3, number of examples)
    Y: train label. 0 if non-cat and 1 if cat. shape (1, number of examples)
    '''

    m = X.shape[1]

    # forward propagation
    A = sigmoid(np.dot(w.T,X) + b)
    #cost: L(a,y) = -( ylog(a) + (1 - y)log(1-a) ), a 为训练集输出， y 未训练集labels
    cost = ( -1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))

    # backward propagation
    dw = (1 / m) * np.dot(X, (A - Y).T) #d(cost)/d(w) = d(cost) / d(a)  *  d(a) / d(w) = a-y
    db = (1 / m) * np.sum(A - Y)

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())


    grads = {"dw":dw,
             "db":db}

    return grads, cost

def optimize(w,b,X,Y,num_loop,learning_rate,print_cost=False):
    '''
    This function optimize w and b by running a gradient descent algorithm

    num_loop: number of optimization loop
    learning_rate: rate of the gradient descent update rule
    '''

    costs = []

    for i in num_loop:
        grads, cost = propagate(w,b,X,Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b = learning_rate * db

        if i % 100 == 0:
            costs.append(cost)
            pass
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))

    params = {
        "w": w,
        "b": b
    }
    grads = {
        "dw": dw,
        "db": db
    }

    return params, grads, costs

def predict(w,b,X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape([0],1))

    A = sigmoid(np.dot(w.T,x) + b)

    for i in range(A.shape[1]):
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0

    assert(Y_prediction.shape == (1,m))
    return Y_prediction
