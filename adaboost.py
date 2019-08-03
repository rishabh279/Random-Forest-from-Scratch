import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from rf_classification import get_data

class AdaBoost:

    def __init__(self, m):
        self.m = m

    def fit(self, x, y):
        self.models = []
        self.alphas = []

        n, _ =x.shape
        w = np.ones(n) / n

        for m in range(self.m):
            tree = DecisionTreeClassifier(max_depth=1)
            tree.fit(x, y, sample_weight=w)
            p = tree.predict(x)

            err = w.dot(p != y)
            alpha = 0.5 * (np.log(1 - err) - np.log(err))

            w = w * np.exp(-alpha*y*p)
            w = w / w.sum()

            self.models.append(tree)
            self.alphas.append(alpha)

    def predict(self, x):
        n , _ = x.shape
        fx = np.zeros(n)
        for alpha, tree in zip(self.alphas, self.models):
            fx += alpha*tree.predict(x)
        return np.sign(fx), fx

    def score(self, x, y):
        p, fx = self.predict(x)
        l = np.exp(-y*fx).mean()
        return np.mean(p == y), l

if __name__ == '__main__':
    x, y = get_data()
    y[y == 0] = -1
    ntrain = int(0.8*len(x))
    xtrain, ytrain = x[:ntrain], y[:ntrain]
    xtest, ytest = x[ntrain:], y[ntrain:]

    t = 200
    train_errors = np.empty(t)
    test_losses = np.empty(t)
    test_errors = np.empty(t)
    for num_trees in range(t):
        if num_trees == 0:
            train_errors[num_trees] = None
            test_errors[num_trees] = None
            test_losses[num_trees] = None
            continue
        model = AdaBoost(num_trees)
        model.fit(xtrain, ytrain)
        acc, loss = model.score(xtest, ytest)
        acc_train, _ = model.score(xtrain, ytrain)
        train_errors[num_trees] = 1 - acc_train
        test_errors[num_trees] = 1 - acc
        test_losses[num_trees] = loss

        if num_trees == t - 1:
            print('Final train error', 1 - acc_train)
            print('Final test error', 1 - acc)
    plt.plot(test_errors, label='test errors')
    plt.plot(test_losses, label='test losses')
    plt.legend()
    plt.show()

    plt.plot(train_errors, label='train errors')
    plt.plot(test_errors, label='test errors')
    plt.legend()
    plt.show()
