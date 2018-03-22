import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
                      header=None)

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
sc = StandardScaler()

X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# Manual count
# np.set_printoptions(precision=4)
# mean_vecs = []
#
# for label in range(1, 4):
#     mean_vecs.append(np.mean(X_train_std[y_train == label], axis=0))
#     # print('MV %s: %s\n' %(label, mean_vecs[label-1]))
#
# d = 13 # number of features
# S_W = np.zeros((d, d))
#
# for label, mv in zip(range(1, 4), mean_vecs):
#     class_scatter = np.zeros((d, d))
#     for row in X_train_std[y_train == label]:
#         row, mv = row.reshape(d, 1), mv.reshape(d, 1)
#         class_scatter += (row - mv).dot((row - mv).T)
#     S_W += class_scatter
#
# S_B = np.zeros((d, d))
# mean_overall = np.mean(X_train_std, axis=0)
#
# for i, mean_vec in enumerate(mean_vecs):
#     n = X_train[y_train == i + 1, :].shape[0]
#     mean_vec = mean_vec.reshape(d, 1)
#     mean_overall = mean_overall.reshape(d, 1)
#
#     S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
#
# eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
#
# eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
# eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

# This is a chart for cumulative discriminability
# tot = sum(eigen_vals.real)
# discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]
# cum_discr = np.cumsum(discr)
# plt.bar(range(1, 14), discr, alpha=0.5, align='center', label='individual discriminability')
# plt.step(range(1, 14), cum_discr, where='mid', label='cumulative discriminability')
# plt.xlabel('"discriminability ratio"')
# plt.ylabel('Linear discriminants')
# plt.ylim([-0.1, 1.1])
# plt.legend(loc='best')
# plt.show()

def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.6,
                    c=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx],
                    label=cl)

lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)

lr = LogisticRegression()
lr = lr.fit(X_train_lda, y_train)
# plot_decision_regions(X_train_lda, y_train, classifier=lr)
# plt.xlabel('LD 1')
# plt.ylabel('LD 2')
# plt.legend(loc='lower left')
# plt.show()

X_test_lda = lda.transform(X_test_std)
plot_decision_regions(X_test_lda, y_test, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.show()
