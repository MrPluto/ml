#### 4 assumptions that needs to be true and orthogonal
- Fit training set well on cost function
    - use bigger neural network or switching to a better optimization algorithm to improve it
- Fit dev set well on cost function
    - regularization or using bigger training set might help
- Fit test set well on cost function
    - using bigger dev set might help
- performs well in real world
    - if it doesn't perform well, means the dev/test set is not set correctly or the cost function is not evaluating the right thing


#### Using a single number evaluation metric
- how to define a metric to evaluate classifiers
- worry separately about how to do well on this metric

#### Train/Dev/Test data

Guideline
- Choose a dev set and test set to reflect data you expect to get in the future and consider important to da well on
- Set up the size of the test set to give a high confidence in the overall performance of the system
- Test set helps evaluate the performance of the final classifier which could be less 30% of the whole data Set
- The development set has to be enough to evaluate different ideas

#### s
- Get labeled data from humans
- Gain insight from manual error analysis
- better analysis of bias/variance

#### Bayes optimal error

#### Human level performance

- you can fit the training set pretty well
- the training set performance generalizes pretty well to the dev/test set
