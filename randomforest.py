import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from rf_classification import get_data

x, y = get_data()
ntrain = int(0.8*len(x))
xtrain, ytrain = x[:ntrain], y[:ntrain]
xtest, ytest = x[ntrain:], y[ntrain:]


class BaggedTreeClassifier:
  def __init__(self, n_estimators, max_depth=None):
    self.B = n_estimators
    self.max_depth = max_depth

  def fit(self, X, Y):
    N = len(X)
    self.models = []
    for b in range(self.B):
      idx = np.random.choice(N, size=N, replace=True)
      Xb = X[idx]
      Yb = Y[idx]

      model = DecisionTreeClassifier(max_depth=self.max_depth)
      model.fit(Xb, Yb)
      self.models.append(model)

  def predict(self, X):
    # no need to keep a dictionary since we are doing binary classification
    predictions = np.zeros(len(X))
    for model in self.models:
      predictions += model.predict(X)
    return np.round(predictions / self.B)

  def score(self, X, Y):
    P = self.predict(X)
    return np.mean(Y == P)

class RandomForest:

    def __init__(self, n_estimators):
        self.b = n_estimators

    def fit(self, x, y, m=None):
        n, d = x.shape
        if m is None:
            m = int(np.sqrt(d))

        self.models = []
        self.features = []
        for b in range(self.b):
            tree = DecisionTreeClassifier()

            features = np.random.choice(d, size=m, replace=False)

            idx = np.random.choice(n, size=n, replace=True)
            xb = x[idx]
            yb = y[idx]

            tree.fit(xb[:, features], yb)
            self.features.append(features)
            self.models.append(tree)

    def predict(self, x):
        n = len(x)
        p = np.zeros(n)
        for features, tree in zip(self.features, self.models):
            p += tree.predict(x[:, features])

        return np.round(p / self.b)

    def score(self, x, y):
        p = self.predict(x)
        return np.mean(p == y)

T = 500
test_error_prf = np.empty(T)
test_error_rf = np.empty(T)
test_error_bag = np.empty(T)
for num_trees in range(T):
    if num_trees == 0:
        test_error_prf[num_trees] = None
        test_error_rf[num_trees] = None
        test_error_bag[num_trees] = None

    else:
        rf = RandomForestClassifier(n_estimators=num_trees)
        rf.fit(xtrain, ytrain)
        test_error_rf[num_trees] = rf.score(xtest, ytest)

        bg = BaggedTreeClassifier(n_estimators=num_trees)
        bg.fit(xtrain, ytrain)
        test_error_bag[num_trees] = bg.score(xtest, ytest)

        prf = RandomForest(n_estimators=num_trees)
        prf.fit(xtrain, ytrain)
        test_error_prf[num_trees] = prf.score(xtest, ytest)


plt.plot(test_error_rf, label='rf')
plt.plot(test_error_prf, label='pseudo rf')
plt.plot(test_error_bag, label='bag')
plt.legend()
plt.show()