[An analysis of the Regularization between L2 and Dropout in Single hidden layer network](http://uksim.info/isms2016/CD/data/0665a174.pdf)

dropout:  preventing co-adaptations among neurons on the training data.
no neurons depend excessively on other neurons to correct its mistakes and
they must work well with other neurons in a wide variety of different situations

> Combining several learning models and the averaging prediction results can enhance
> the overall performance. (each model should use different training dataset)?

> The objective of randomly dropping out neurons is to allow
> each neuron to learn something userful on its own without relying too much on other
> neurons to correct its shortcomings.
>
> like 滥竽充数


#### MNIST dataset with random initialization of weight parameters
L2: better in **small network** performance will **significantly decreased in large network**

dropout: performance will also reduced in large network but the **rate of change is slightly slower
and more rubust than L2**

#### MNIST dataset with pretraining of weight parameters ([SAE](http://ufldl.stanford.edu/wiki/index.php/%E6%A0%88%E5%BC%8F%E8%87%AA%E7%BC%96%E7%A0%81%E7%AE%97%E6%B3%95))

performance can be improved by pretraining parameters with SAE,[RBM](https://deeplearning4j.org/cn/restrictedboltzmannmachine)

dropout: has more room of improvement and slowly drops the performance when the network
 is more complex.
