# E-swish: A New Novel Activation Function
Code for reproducibility of [this paper]: (https://google.com)

## Formula
<div style="text-align:center"><img src ="https://github.com/EricAlcaide/E-swish/blob/master/README.md/e_swish.png" /></div>

## Abstract

Activation functions have a notorious impact in neural networks on both training and testing the models against the desired problem. Currently, the most used activation function is the Rectified Linear Unit (ReLU). Although some alternatives have been proposed, none of them have managed to replace ReLU as the default activation due to inconstant improvements. This paper introduces a new activation function, closely related with the new activation Swish = x * sigmoid (x) (Ramachandran et al., 2017) which we call E-swish.
E-swish is just a Swish activation function mirrored across the identity for all positive values. We show that E-swish outperforms many other well-known activations on a variety of tasks and it also leads to a faster convergence. For example, replacing Relu by E-swish provided at 0.3% accuracy improvement on Cifar10 for the WRN 16-4.

## Results

### MNIST

| Activations      | % Accuracy (median of 3 runs) |
| -------------    | ------------- |
| E-swish          | 98.46         |
| Swish            | 98.31         |
| Relu             | 98.30         |
| Elu              | 97.92         |
| Leaky Relu (0.3) | 97.71         |
### CIFAR10

* **Simple CNN**

| Activations      | % Accuracy (median of 3 runs) |
| -------------    | ------------- |
| E-swish          | 83.35         |
| Swish            | 81.88         |
| Relu             | 82.06         |
| Elu              | 81.81         |
| Leaky Relu (0.3) | 82.22         |

* **Deeper CNN**

| Activations      | % Test error (median of 3 runs) |
| -------------    | ------------- |
| E-swish          | 7.61          |
| Swish            | 8.27          |
| Relu             | 8.13          |
| Elu              | 8.95*         |
| Leaky Relu (0.3) | 9.25*         |

<i>*: Results are provided on a single model performance.</i>

* **WRN 16-4**

| Activations      | % Test error (median of 5 runs) |
| -------------    | ------------- |
| Relu             | 5.02          |
| E-swish          | 4.71          |

## Contact

* **Email:** ericalcaide1@gmail.com
* **Twitter:** https://twitter.com/eric_alcaide