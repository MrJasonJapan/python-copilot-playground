# import numpy and pyplot
import numpy as np
import matplotlib.pyplot as plt


# create a linspace array from 0 to 2*np.pi with 100 points
x = np.linspace(0, 2*np.pi, 100)

# create y based on sin of x
y = np.sin(x)

# plot x and y into a file
plt.plot(x, y)
plt.savefig('sin.png')

# create two 1D arrays with 2 points each
x = np.array([0, 1])
y = np.array([0, 1])

# take the dot product of x and y
print(np.dot(x, y))

