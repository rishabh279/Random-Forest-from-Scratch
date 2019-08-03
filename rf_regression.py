import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

NUMERICAL_COLS = [
    'crim',
    'zn',
    'nonretail',
    'nox',
    'rooms',
    'age',
    'dis',
    'rad',
    'tax',
    'ptratio',
    'b',
    'lstat',
]

NO_TRANSFORM = ['river']


class DataTransformer:

    def fit(self, df):
        self.scalers = {}
        for col in NUMERICAL_COLS:
            scaler = StandardScaler()
            scaler.fit(df[col].values.reshape(-1,1))
            self.scalers[col] = scaler

    def transform(self, df):
        n, _ = df.shape
        d = len(NUMERICAL_COLS) + len(NO_TRANSFORM)
        x = np.zeros((n, d))
        i = 0
        for col, scaler in self.scalers.items():
            x[:, i] = scaler.transform(df[col].values.reshape(-1, 1)).flatten()
            i += 1
        for col in NO_TRANSFORM:
            x[:, i] = df[col]
            i += 1
        return x

    def fit_transformer(self, df):
        self.fit(df)
        return self.transform(df)


def get_data():
    df = pd.read_csv('data/housing.data', header=None, sep=r"\s*", engine='python')
    df.columns = [
        'crim',  # numerical
        'zn',  # numerical
        'nonretail',  # numerical
        'river',  # binary
        'nox',  # numerical
        'rooms',  # numerical
        'age',  # numerical
        'dis',  # numerical
        'rad',  # numerical
        'tax',  # numerical
        'ptratio',  # numerical
        'b',  # numerical
        'lstat',  # numerical
        'medv',  # numerical -- this is the target
    ]

    transformer = DataTransformer()
    n = len(df)
    train_idx = np.random.choice(n, size=int(0.7*n), replace=False)
    test_idx = [i for i in range(n) if i not in train_idx]
    df_train = df.loc[train_idx]
    df_test = df.loc[test_idx]

    xtrain = transformer.fit_transformer(df_train)
    ytrain = np.log(df_train['medv'].values)
    xtest = transformer.transform(df_test)
    ytest = np.log(df_test['medv'].values)
    return xtrain, ytrain, xtest, ytest


if __name__ == '__main__':
    xtrain, ytrain, xtest, ytest = get_data()

    model = RandomForestRegressor(n_estimators=100)
    model.fit(xtrain, ytrain)
    predictions = model.predict(xtest)

    plt.scatter(ytest, predictions)
    plt.xlabel("target")
    plt.ylabel("prediction")
    ymin = np.round(min(min(ytest), min(predictions)))
    ymax = np.ceil(max(max(ytest), max(predictions)))
    r = range(int(ymin), int(ymax) + 1)
    plt.plot(r, r)
    plt.show()

    plt.plot(ytest, label='targets')
    plt.plot(predictions, label='predictions')
    plt.legend()
    plt.show()