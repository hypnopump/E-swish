import numpy as np 
import matplotlib.pyplot as plt 

def sigmoid(x):
	print(type(x))
	return 1/(1+np.exp(-x))

x = np.linspace(-6, 6, 1000)

def e_swish_deriv(x):
	x = list(x)
	mods = []

	for i, it in enumerate(x):
		if it>(0-0.01) and it<(0+0.01):
			mods.append(i)
		if it>0:
			x[i] = 2 - (it*sigmoid(it)+sigmoid(it)*(1-it*sigmoid(it)))
		else:
			x[i] = it*sigmoid(it)+sigmoid(it)*(1-it*sigmoid(it))

	x = np.array(x)
	print(mods, x.shape)
	x[mods[0]:mods[-1]] = np.nan
	return x

def e_swish_second_deriv(x):
	x = list(x)
	mods = []

	for i, it in enumerate(x):
		first = (2+it)*np.exp(-2*it)
		num = (first+2*np.exp(-it)-it*np.exp(-it))
		den = (np.exp(-it)+1)**3
		
		if it>(0-0.01) and it<(0+0.01):
			mods.append(i)
		if it>0:
			x[i] = -num/den
		else:
			x[i] = num/den

	x = np.array(x)
	print(mods, x.shape)
	x[mods[0]:mods[-1]] = np.nan

	return x

def plot(x):
	first = e_swish_deriv(x)
	second = e_swish_second_deriv(x)

	plt.figure()
	plt.title("E-swish first and second derivatives")
	plt.xlim(-6, 6)
	plt.plot(x, first, 'r-', linewidth=2)
	plt.plot(x, second, '--', linewidth=2)
	plt.grid()
	plt.legend(["First derivative", "Second derivative"], loc="upper left")
	plt.show()
	return

# plot(x)

def plot_2(x):
	# x = np.linspace(-5, 3, 1000)
	first = x*sigmoid(x)
	second = x*(2-sigmoid(x))

	y = np.maximum(first, second)
	y_relu = np.maximum(0, x)
	y_swish = x*sigmoid(x)
	y_elu = np.maximum(np.minimum(0,np.exp(x)-1), x)
	y_leaky = np.maximum(0.3*x, x)

	plt.figure()
	plt.title("E-swish and other well-know activations")
	plt.ylim(-5, 3)

	plt.plot(x, y, '-')

	plt.plot(x, y_swish, '-')
	plt.plot(x, y_relu, '-')
	plt.plot(x, y_elu, '-')
	plt.plot(x, y_leaky, '-')

	plt.grid()
	plt.legend(["E-swish", "Swish", "Relu", "Elu", "Leaky Relu"], loc="upper left")
	plt.show()
	return

x = np.linspace(-5, 3, 1000)
plot_2(x)