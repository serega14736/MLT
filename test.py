import numpy as np
# from math import exp
import matplotlib.pyplot as plt
# x = np.arange(-4.0, 4.0, 0.001)
# y = np.empty(0)
# z = np.empty(0)	
# for xi in x:	
	# z = np.append(z, pow(1-xi*xi, 2) * int(abs(xi) < 1))
	# y = np.append(y, exp((-1/2) * xi*xi))
# plt.plot(x, y)
# plt.plot(x, z)

# plt.legend ( ("гауссовское ядро", "квартичесое ядро") )
# plt.show()



from sklearn.datasets import load_boston as data

X = data().data[:, np.newaxis, 5]
y = data().target
plt.ylabel('Средняя стоимость домов')
plt.xlabel('Cреднее количество комнат в доме')
plt.plot(X, y, 'o', markersize=2.5, color='black')
plt.show()