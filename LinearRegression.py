import numpy as np
import matplotlib.pyplot as plt

#X.shape = (m,n), X1.shape=(m,n+1), params.shape=(n+1,1), y.shape=(m,1)

class LinearRegression:
    def __init__(self):
        self.params = None
        self.errors = []
    
    def calculate_error(self, y, y_pred):
        m = y.shape[0]
        error = 1/(2*m)*np.sum(np.square(y-y_pred))
        return error
    
    def plot_error(self):
        fig, ax = plt.subplots()
        ax.plot(self.errors)
        plt.show()

    def predict(self, X):
        m = X.shape[0]
        a = np.ones((X.shape[0],1))
        X1 = np.concatenate([a, X], axis=1)
        y_hat = np.matmul(X1, self.params)
        return y_hat
    
    def fit(self, X, y, learning_rate=0.01, num_iter=1000):
        self.errors = []
        m = X.shape[0]
        n = X.shape[1]
        self.params = np.random.rand(n+1,1)
        a = np.ones((X.shape[0],1))
        X1 = np.concatenate([a, X], axis=1)
        for _ in range(num_iter):
            y_hat = self.predict(X)
            self.params = self.params - learning_rate*(1/m)*np.matmul(X1.T, (y_hat-y))
            self.errors.append(self.calculate_error(y, y_hat))

if __name__ == "__main__":
    X = np.random.rand(1000,200)
    y = np.random.rand(1000,1)
    l = LinearRegression()
    l.fit(X, y)
    print(l.predict(X))
