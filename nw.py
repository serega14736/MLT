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

def nadarayaWatson (X, y, kernel, metric, h):		
	n = X.size
	Y = np.zeros(n)
	for l in range(n):
		W = np.zeros(n)
		for i in range(n):
			W[i] = kernel(metric(X[l], X[i]) / h)
		Y[l] = sum(W * y) / sum(W)
	return Y
	
def SSE(y, Y):
    return ((Y - y) ** 2).sum()

def generate_wave_set(n_support=1000, n_train=250):
    data = {}
    data['support'] = np.linspace(0, 2 * np.pi, num=n_support)
    data['x_train'] = np.sort(np.random.choice(data['support'], size=n_train, replace=True))
    data['y_train'] = np.cos(data['x_train']).ravel()
    data['y_train'] += 0.5 * (0.55 - np.random.rand(data['y_train'].size))
    return data
	
data = generate_wave_set(100, 80)
X = data['x_train']
y = data['y_train']

SSEsG = np.empty(0)
SSEsQ = np.empty(0)

for h, color in zip(H, colors):
	Y = nadarayaWatson(X, y, kernelG, euclidean, h)
	plt.plot(X, Y, color=color, label='h = ' + str(format(h, '.1f')))
	
	SSEsG = np.append(SSEsG, SSE(y, Y))
plt.title('Ядерное сглаживание с гауссовским ядром')
plt.plot(X, y, 'o', markersize=2.5, color='black')
plt.legend()
plt.show()

for h, color in zip(H, colors):
	Y = nadarayaWatson(X, y, kernelQ, euclidean, h)
	plt.plot(X, Y, color=color, label='h = ' + str(format(h, '.1f')))
	
	SSEsQ = np.append(SSEsQ, SSE(y, Y))
plt.title('Ядерное сглаживание с квартическим ядром')
plt.plot(X, y, 'o', markersize=2.5, color='black')
plt.legend()
plt.show()

dataF = {'h':H,'гауссовское ядро':SSEsG, 'квартическое ядро': SSEsQ}
df = pd.DataFrame(dataF)
df.style