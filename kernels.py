import torch
import matplotlib.pyplot as plt

def epn(x):
    return (5/(2*torch.pi)) * (1 - x**2)

def tricube(x):
    return (70/81) * (1 - torch.abs(x)**3) ** 3

def triweight(x):
    return (1 - x**2) ** 3

def fc(x):
    return 1/x

x = torch.linspace(-1, 1, 100, dtype=torch.float32)

plt.figure()
plt.plot(epn(x))
plt.show()

