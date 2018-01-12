[总结](http://blog.csdn.net/Asun0204/article/details/78131609)

[Overview](http://ruder.io/optimizing-gradient-descent/index.html)

**What you should remember**:
- The difference between Gradient Descent, mini-batch Gradient Descent and stochastic Gradient Descent is the number of examples you use to perform one update step.
- You have to tune a learning rate hyperparameter **a(阿尔法)**
- With a well-turned mini-batch size, usually it outperforms either Gradient Descent or stochastic Gradient Descent (particularly when the training set is large).

### Gradient Descent
#### mini-batch Gradient Descent
if mini-batch size = m, => batch Gradient Descent
if mini-batch size = 1, => stochastic Gradient Descent
if mini-batch size between (1,m), =>

##### notes
- Shuffling and Partitioning are the two steps required to build mini_batches
- Powers of two are often chosen to be the mini-batch size

##### features
<!-- - not need to wait one epoch finished -->
- fast

#### SGD (stochastic Gradient Descent)
##### 3 for-loops in total
- Over the number of iterations
- Over the m training examples
- Over the layers (update parameters)

##### features
- never converge, always kind of oscillate and wander around the region of the minimum
- lose all your speed up from vertorization ( matrix calculate )
- decrease the learning_rate slowly,SGD shows the same convergence behavior like batch GD

### Optimization method
#### Momentum

> The momentum term increases for dimensions whose gradients point in the same directions and reduces updates for dimensions whose gradients change directions. As a result, we gain faster convergence and reduced oscillation

##### Formula
V(dW)t = beta1 * V(dW)t + (1 - beta1) * dW

W = W - learning_rate * V(dW)t

V(db)t = beta1 * V(db)t + (1 - beta1) * db

b = b - learning_rate * V(db)t

##### notes
- The velocity is initialized with zeros. So the algorithm will take a few iterations to "build up" velocity and start to take bigger steps. (or see **V(dW)t_corrected = V(dW)t / (1 - beta ^ t)**)
- `beta` = `0`, means normal Graident Descent
- the large beta, the smoother the update. (about 1/(1-beta), see Exponentially Weighted Averages)


#### RMSprop
##### Formula
S(dW)t = beta2 * S(dW)t + (1 - beta2) * dW^2

W = W - learning_rate * dW / sqrt(S(dW)t + e)

S(db)t = beta2 * S(db)t + (1 - beta2) * db^2

b = b - learning_rate * db / sqrt(S(db)t + e)

e ≈ 1e-8  (suggested)

#### Adam

##### Formula
V(dW)t = beta1 * V(dW)t + (1 - beta1) * dW

V(dW)t_corrected = V(dW)t / (1 - beta1 ^ t)

V(db)t = beta1 * V(db)t + (1 - beta1) * db

V(db)t_corrected = V(db)t / (1 - beta1 ^ t)

S(dW)t = beta2 * S(dW)t + (1 - beta2) * dW^2

S(db)t = beta2 * S(db)t + (1 - beta2) * db^2

W = W - learning_rate * V(dW)t_corrected / sqrt(S(dW)t + e)

b = b - learning_rate * V(db)t_corrected / sqrt(S(db)t + e)

beta1 -> 0.9

beta2 -> 0.999

e ≈ 10 ^ (-8)  (avoid dividint by zero)

#### Conclusion
- make sure mini-batch fit the CPU/GPU memory
- The difference between gradient descent, mini-batch gradient descent and stochastic gradient descent is the number of examples you use to perform one update step.
- You have to tune a learning rate hyperparameter  αα .
- With a well-turned mini-batch size, usually it outperforms either gradient descent or stochastic gradient descent (particularly when the training set is large).
