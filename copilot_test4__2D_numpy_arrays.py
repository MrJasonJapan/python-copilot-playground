# import numpy
import numpy as np

# create a 3x3 numpy array 
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# print the dimensions of the array
print(a.ndim)

# print the shape of the array
print(a.shape)

# print the size of the array
print(a.size)

# print the element in the 2nd row and 3rd column
print(a[1, 2])

# slice the first two columns from the first two rows and assign to new variable
b = a[:2, :2]
print(b)

# print an empty line
print()

# create two 2x2 arrays
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# add the two matrices and assign to c
c = a + b
print(c)

# print an empty line
print()

# multiply c by 2
c *= 2
print(c)

# print an empty line
print()

# take the hadamard product of a and b and assign to d
d = a * b
print(d)

# print an empty line
print()

# define two simple 2D arrays (2x3 & 3x2) to use for matrix multiplication
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[1, 2], [3, 4], [5, 6]])

# take the dotproduct of a and b and assign to c
f = np.dot(a, b)
print(f)










