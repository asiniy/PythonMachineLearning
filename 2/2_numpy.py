# Taken from this one: https://sebastianraschka.com/pdf/books/dlb/appendix_f_numpy-intro.pdf

import numpy as np

# list = [
#     [1, 2, 3],
#     [4, 5, 6]
# ]
#
# ary1d = np.array(list)
#
# m, n = ary1d.shape
# print(m)
# print(n)

# print(np.ones((4, 7)))
# print(np.zeros((3, 5)))

# print(np.eye(3))
# print(np.diag((3, 9, 8)))

# print(np.arange(4, 10))
# print(np.arange(4., 10.))
# print(np.arange(5))
# print(np.arange(5.))
# print(np.arange(1., 11., 2))

# print(np.linspace(0., 1., num=7))

# print(ary[0, 0])
# print(ary[-1, -1])
# print(ary[1, 1])

# print(ary[0])
# print(ary[:, 1])
# print(ary[:, :2])

# print(ary ** 3)

# print(np.add.reduce(ary))
# print(np.add.reduce(ary, axis=1))

# print(np.sum(ary, axis=0))
# print(np.sum(ary, axis=1))
# print(np.sum(ary))


# print(ary[0].__class__)
# print(ary.__class__)

# print(np.argmin(ary))
# print(np.argmax(ary))

# ary_1 = np.array([1, 2, 3])
# print(ary + np.array([1, 2]))

ary = np.array([[1, 2, 3], [4, 5, 6]])

print(ary.T)
