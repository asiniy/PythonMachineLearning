import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# s = pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])
# print(s[2:] + s[:-1])

# d = {
#     'one': pd.Series([1., 2., 3.], index=['a', 'b', 'c']),
#     'two': pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])
# }
#
# df = pd.DataFrame(d, columns=['one', 'two'])

# iris = pd.read_csv('iris.data')
# (iris
#     .query('SepalLength > 7')
#     .assign(SepalRatio = lambda x: x.SepalWidth / x.SepalLength,
#             PetalRatio = lambda x: x.PetalWidth / x.PetalLength)
#     .plot(kind='scatter', x='SepalRatio', y='PetalRatio'))
#
# plt.show()

# s = pd.Series([1, 3, 5, np.nan, 6, 8])

# s = pd.date_range('20130101', periods=6)

# df = pd.DataFrame(np.random.randn(1000000, 4), columns=list('ABCD'))

s = pd.Series(["a", "b", "c", "a"], dtype="category")

print(s)
