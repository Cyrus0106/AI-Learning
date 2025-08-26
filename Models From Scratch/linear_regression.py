import numpy as np


class SimpleLinearRegression:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        # convert to numpy
        X = np.array(X)
        y = np.array(y)

        # columns of 1s if it intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0], X)]

        # Normal Equation w = (X^T X)^-1 X^T y
        w = np.linalg.pinv(X.T @ X) @ (X.T @ y)

        if self.fit_intercept:
            self.intercept_ = w[0]
            self.coef_ = w[1:]
        else:
            self.intercept_ = 0
            self.coef_ = w

    def predict(self, X):
        X = np.array(X)
        intercept = self.intercept_ if self.intercept_ is not None else 0
        return X @ self.coef_ + intercept


class LinearRegressionGD:
    def __init__(self, lr=0.01, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.coef = None
        self.intercept_ = 0

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        # initialise weights
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0

        # Gradient Descent
        for _ in range(self.n_iter):
            y_pred = X @ self.coef_ + self.intercept_
            error = y_pred - y
            # gradients
            dw = (2 / n_samples) * (X.T @ error)
            db = (2/n_samples) * np.sum(error)

            # update weights
            self.coef_ -= self.lr * dw
            self.intercept_ -= self.lr * db

    def predict(self, X):
        return np.array(X) @ self.coef_ + self.intercept_