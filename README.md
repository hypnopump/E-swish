# E-swish: Adjusting Activations to Different Network Depths
This repository will contain the code for reproducibility of the experiments of [this paper](https://arxiv.org/abs/1801.07145v1)

Uploading soon.

## Formula
<div style="text-align:center"><img src ="e_swish.PNG" /></div>

## Abstract

<p align="justify">
	Activation functions have a notorious impact on neural networks on both training and testing the models against the desired problem. Currently, the most used activation function is the Rectified Linear Unit (ReLU). This paper introduces a new and novel activation function, closely related with the new activation <i><b>Swish=x∗sigmoid(x)</b></i> (Ramachandran et al., 2017) which generalizes it. We call the new activation <i><b>E−swish=βx∗sigmoid(x)</b></i> . 
	We show that E-swish outperforms many other well-known activations including both ReLU and Swish. For example, using E-swish provided 1.5% and 4.6% accuracy improvements on Cifar10 and Cifar100 respectively for the WRN 10-2 when compared to ReLU and 0.35% and 0.6% respectively when compared to Swish. The code to reproduce all our experiments can be found at https://github.com/EricAlcaide/E-swish
</p>

## Results

### CIFAR10

* **WRN 10-2**

| Activations         		| % Accuracy (median of 3 runs) |
| -------------    			| -------------:|
| Relu            			| 89.98         |
| Swish            			| 91.52         |
| **E-swish (beta=1.375)**  | **91.89**     |

* **SimpleNet**

| Activations         		| % Accuracy (single model performance) |
| -------------    			| -------------:|
| Relu            			| 95.33         |
| Swish            			| 95.76         |
| **E-swish (beta=1.125)**  | **96.02**     |
| E-swish (beta=1.25)       | 95.73         |


### CIFAR100

* **WRN 10-2**

| Activations         		| % Accuracy (median of 3 runs) |
| -------------    			| -------------:|
| Relu            			| 64.27         |
| Swish            			| 68.32         |
| **E-swish (beta=1.375)**  | **68.91**     |

* **WRN 16-4**

| Activations         		| % Accuracy (single model performance) |
| -------------    			| -------------:|
| Relu            			| 75.92         |
| Swish            			| 76.88         |
| **E-swish (beta=1.25)**   | **77.35**     |
| E-swish (beta=1.5)        | 77.25         |
| E-swish (beta=1.75)       | 77.29         |
| E-swish (beta=2)          | 77.14         |

## Contact

* **Email:** ericalcaide1@gmail.com
* **Twitter:** https://twitter.com/eric_alcaide