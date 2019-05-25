import numpy as np

# example 1:
data1 = np.ones((3,3))
arr2 = np.array(data1)
arr3 = np.asarray(data1)
data1[1]= 2
print('data1:\n', data1)
print('arr2:\n', arr2)
print('arr3:\n', arr3)