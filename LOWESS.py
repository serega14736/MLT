import numpy as np
import matplotlib.pyplot as plt
from math import exp, sqrt
import pandas as pd

H = np.arange(0.1, 1.0, 0.1)
colors = ['red', 'green', 'blue', 'violet', 'yellow', 'brown', 'orange', 'gray', 'magenta']
kernels = ['гауссовское', 'квартическое']

def kernelG (x):
	return exp((-1 / 2) * x * x)
	
def kernelQ (x):
	return pow(1-x*x, 2) * int(abs(x) < 1)

def euclidean (x, xi):
	return sqrt(pow(x-xi, 2))

def nadarayaWatson (X, y, kernel, metric, h, i, gamma):
	n = X.size
	W = np.zeros(n)
	a = 0
	b = 0
	for j in range(n):
		if(j != i):
			W[j] = kernel(metric(X[i], X[j]) / h)
	for j in range(n):
		if(j != i):
			a = a + W[j] * y[j] * gamma[j]
			b = b + W[j] * gamma[j]
	return a / b

def LOO(X, y, kernel, metric, h, gamma):
	n = X.size
	a = np.zeros(n)
	for i in range(n):
		a[i] = nadarayaWatson(X, y, kernel, metric, h, i, gamma)
	return a

def errors_values(X, y, kernel, metric, h, CV, gamma):
	n = X.size
	eps = np.zeros(n)
	a = CV(X, y, kernel, metric, h, gamma)
	for i in range(n):
		eps[i] = abs(a[i] - y[i])
	return eps

def med(eps):
    if len(eps)%2==0:
        eps = sorted(eps)
        num = round(len(eps)/2)
        num2 = num-1
        middlenum = (eps[num]+eps[num2])/2
    else:
        eps = sorted(eps)
        listlength = len(eps) 
        num = round(listlength / 2)
        middlenum = eps[num]
    return middlenum

def lowess_kernelQ(eps, i):
	return kernelQ(eps[i] / (6 * med(eps)))

def LOWESS(X, y, kernel, metric, h, lowess_kernel, errors, CV, max_iter):
	n = X.size
	gamma = np.ones(n)
	for j in range(max_iter):
		eps = errors(X, y, kernel, metric, h, CV, gamma)
		for i in range(n):
			gamma[i] = lowess_kernel(eps, i)
	return CV(X, y, kernel, metric, h, gamma)

	
# def SSE(y, Y):
#     return ((Y - y) ** 2).sum()

def generate_wave_set(n_support=1000, n_train=250):
    data = {}
    data['support'] = np.linspace(0, 2 * np.pi, num=n_support)
    data['x_train'] = np.sort(np.random.choice(data['support'], size=n_train, replace=True))
    data['y_train'] = np.cos(data['x_train']).ravel()
    data['y_train'] += 0.5 * (0.55 - np.random.rand(data['y_train'].size))
    return data

data = generate_wave_set(100, 80)
_X = data['x_train']
_y = data['y_train']

_X[12] = 3
_y[12] = 0

# _a = LOWESS(_X, _y, kernelG, euclidean, 0.5, lowess_kernelQ, errors_values, LOO, 1175)

# plt.plot(_X, _y, 'o', markersize=2.5, color='black')

# plt.plot(_X, _a)
# plt.show()

step = (_X.max() - _X.min()) / _X.shape[0]
_X_test = np.arange(_X.min(), _X.max(), step)
print(_X_test.size)