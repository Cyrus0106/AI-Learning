import numpy as np


class LogisticRegression:
    def __init__(self, lr=0.001, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.intercept_ = 0
        self.coef_ = None

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def cross_entropy(self, y, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return (-1 / len(y)) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        self.intercept_ = 0
        self.coef_ = np.zeros(n_features)

        for _ in range(self.n_iter):
            z = self.coef_ @ X + self.intercept_
            y_pred = self.sigmoid(z)
            cost = self.cross_entropy(y, y_pred)
            error = y_pred - y
            dw = (1 / n_samples) * (X.T @ error)
            db = (1 / n_samples) * np.sum(error)

            self.coef_ -= self.lr * dw
            self.intercept_ -= self.lr * db

    def predict_proba(self,X):
        z = np.array(X) @ self.coef_ + self.intercept_
        return self.sigmoid(z)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities > 0.5).astype(int)
