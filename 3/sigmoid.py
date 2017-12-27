import matplotlib.pyplot as plt
import numpy as np

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def cost_1(z):
    return -np.log(sigmoid(z))

def cost_0(z):
    return -np.log(1 - sigmoid(z))

z = np.arange(-10, 10, 0.1)

phi_z = sigmoid(z)
c_1 = [cost_1(x) for x in z]
plt.plot(phi_z, c_1, label='J(w) if y=1')

c_0 = [cost_0(x) for x in z]
plt.plot(phi_z, c_0, label='J(w) if y=0', linestyle='--')

plt.ylim(0.0, 5.1)
plt.xlim([0, 1])
plt.xlabel('$\phi$(z)')
plt.ylabel('J(w)')
plt.legend(loc='best')
plt.show()
