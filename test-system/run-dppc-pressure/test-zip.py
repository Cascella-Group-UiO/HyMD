import numpy as np

A = tuple(np.array([[1,2],[3,4]]))

B = tuple(np.array([[5,6],[7,8]]))

res = [sum(n) for n in zip(A,B)]

#print(res)
