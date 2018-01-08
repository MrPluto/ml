**What you should remember**:
- The difference between Gradient Descent, mini-batch Gradient Descent and stochastic Gradient Descent is the number of examples you use to perform one update step.
- You have to tune a learning rate hyperparameter **a(阿尔法)**
- With a well-turned mini-batch size, usually it outperforms either Gradient Descent or stochastic Gradient Descent (particularly when the training set is large).

####batch Gradient Descent

####mini-batch Gradient Descent
if mini-batch size = m, => batch Gradient Descent
if mini-batch size = 1, => stochastic Gradient Descent
if mini-batch size between (1,m), =>

- not need to wait the epoch finished
- fast

####SGD (stochastic Gradient Descent)
- never converge, always kind of oscillate and wander around the region of the minimum
- lose all your speed up from vertorization ( matrix calculate )

####

####Conclusion
- if training set size <=2000, batch Gradient Descent
- typical mini-batch size, 2^6,2^7,2^8,2^9
- make sure mini-batch fit the CPU/GPU memory
