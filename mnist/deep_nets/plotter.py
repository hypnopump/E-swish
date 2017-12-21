import numpy as np
import matplotlib.pyplot as plt 

n_layers = major_ticks = np.arange(23, 45, 3) 

arr = np.array([
	[0.9717, 0.9253, 0.8639, 0.6807, 0.5834, 0.1135, 0.1572, 0.1152], 			# relu
	[0.9753, 0.9682, 0.9487, 0.9353, 0.9047, 0.7936, 0.6790, 0.6189], 	# e-swish
	[0.9855, 0.9771, 0.9682, 0.9467, 0.8342, 0.7813, 0.6553, 0.5968]  # swish
	])

fig, ax = plt.subplots()
for item in arr:
    ax.plot([x for x in range(23,45,3)], [y for y in item], '-o')

plt.grid()
plt.legend(["relu", "e-swish", "swish"], loc='upper right')
ax.set_ylim(0,1)
plt.title("Training deep nets on MNIST")
plt.ylabel("Test accuracy (median of 3 runs)")
plt.xlabel("Number of layers")
plt.xticks(range(23,45,3))
plt.show()
