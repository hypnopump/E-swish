# Plot a provisional graph
import numpy as np 
import matplotlib.pyplot as plt

relu = np.array([
		40.41, 62.04, 65.16, 75.06, 73.78, 79.12, 77.24, 81.24, 83.04, 84.86,
		83.44, 85.38, 85.37, 86.05, 85.64, 86.92, 88.00, 87.67, 86.44, 87.95,
		85.92, 88.37, 87.31, 86.18, 90.00, 89.08, 88.60, 88.35, 89.34, 89.95,
		88.76, 88.85, 90.18, 90.19, 89.34, 89.34, 89.09, 90.53, 90.06, 90.87,
		90.98, 90.54, 90.61, 90.28, 90.70, 90.79, 90.42, 91.60, 90.73, 90.88,
		92.12, 91.95, 92.28, 92.62, 92.06, 91.98, 92.08, 91.88, 92.44, 92.00,
		92.21, 92.05, 92.29, 92.08, 92.03, 91.93, 92.21, 92.19, 92.37, 92.77,
		92.37, 92.10, 92.40, 92.34, 92.57, 92.28, 92.27, 92.49, 92.66, 92.13,
		92.67, 92.80, 92.81, 92.44, 92.71, 92.51, 92.33, 92.50, 92.60, 92.90,
		93.04, 93.08, 92.68, 92.72, 92.68, 92.94, 92.27, 92.86, 92.37, 92.48,
		92.71, 92.94, 92.80, 92.68, 92.40
	])

e_swish_2 = np.array([

	])

swish = np.array([

	])

e_swish_1 = np.array([

	])

content_test =  [e_swish_1, e_swish_2, swish, relu]
labels_test =  ["e_swish_1", "e_swish_2", "swish", "relu"]


# Training
plt.figure()
plt.title("Accuracy of different activations on CIFAR-10 SimpleNet")
plt.xlabel("Number of epochs")
plt.ylabel("Accurary(%) at test time")
plt.grid()
for thing in content_test:
	plt.plot(thing, "-")

plt.legend(labels_test, loc="lower right")
# plt.ylim((85,92))
# plt.xlim((60,190))
plt.show()

