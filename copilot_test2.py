# completely totally funky!!

# import numpy
import numpy as np

# create a numpy array from 0 to 4
a = np.array([0, 1, 2, 3, 4])

# print array type
print(type(a))

# print the array data type
print(a.dtype)

# print the array size
print(a.size)

# print the shape of the array
print(a.shape)

# change the first value of the array to 100
a[0] = 100

# change the array data type to float
a = a.astype(np.float)

# using a for loop, add 3.69 to all values in the array
for i in range(a.size):
    a[i] += 3.69

# print the array
print(a)

# using data comprehension, add 3.69 to all values in the array
a = [i + 3.69 for i in a]

# print the array
print(a)






