import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import ListedColormap

def newline(p1, p2):
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()

    if(p2[0] == p1[0]):
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
        ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])

    l = mlines.Line2D([xmin,xmax], [ymin,ymax])
    ax.add_line(l)
    return l

def plot_decision_regions(X, y, classifier, resolution = 0.02):
    dot1_y = -(classifier.w_[0] + classifier.w_[1] * 1.) / classifier.w_[2]
    dot2_y = -(classifier.w_[0] + classifier.w_[1] * 5.) / classifier.w_[2]

    newline([1, dot1_y], [5, dot2_y])

    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')

    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    plt.show()

class Perceptron(object):
    """
    Perceptron classifier.

    Parameters:
    ---
    eta: float
        Learning rate (between 0.0 and 1.0)
    n_iter: int
        Passes over the training dataset
    random_state: int
        Random number generator seed for random weight initialization

    Attributes:
    ---
    w_ : 1d-array
        Weights after fitting
    errors_: list
        Number of misclassifications in each epoch
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit training data.

        Parameters:
        ---
        X : {array-like}, shape=[n_samples, n_features]
            Training vectors, where n_samples is the number of samples and n_features in the number of features
        y : {array-like}, shape=[n_samples]
            Target values

        Returns:
        ---
        self : object
        """
        rgen = np.random.RandomState(self.random_state)

        self.w_ = rgen.normal(loc = 0.0, scale=0.01, size=1 + X.shape[1])

        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            print('---')

            plot_decision_regions(X, y, self)
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            print(self.w_)
            print('errors:', errors)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)

df = pd.read_csv('2/iris.data')

y = df.iloc[0:100, 4].values
y = np.where(y == 'setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values

ppn = Perceptron(eta = 0.8, n_iter = 6)
ppn.fit(X, y)
