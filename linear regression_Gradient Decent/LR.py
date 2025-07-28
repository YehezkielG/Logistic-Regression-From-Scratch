import numpy as np

class Model():
    def __init__(self, learning_rate=0.001,epoch=0, tol=1e-9):
        self.weight = 0
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.tol = tol
        self.bias = 0
        pass
    
    def fit(self, X, y):
        m = X.shape[1]
        self.weight = np.random.uniform(-0.01, 0.01, size=m)
        prev_cost = float('inf');
        cost = 1
        iteration = 1
        while iteration <= self.epoch if self.epoch else abs(prev_cost-cost) > self.tol:
            prev_cost = cost
            y_hat = self.predict(X)
            cost = self.lost_func(X,y,y_hat)
            self.gradient_decent(X,y,y_hat)
            iteration += 1;
    
    def gradient_decent(self,X:np.ndarray,y:np.ndarray,y_hat):
        m = X.shape[1]
        # deriv = 1/m * np.dot(X.T, (y_hat-y))
        dw = (1/m) * np.dot(X.T, (y_hat - y))
        db = (1/m) * np.sum(y_hat - y)
        self.weight -= self.learning_rate * dw
        self.bias -= self.learning_rate * db
    
    def predict(self,X:np.ndarray):
        return np.dot(X, self.weight) + self.bias
    
    def lost_func(self,X:np.ndarray, y:np.ndarray,y_hat):
        m = X.shape[1]
        return 1/m * (np.sum(y_hat - y)**2)