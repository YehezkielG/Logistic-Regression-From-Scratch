import numpy as np

class model:
    def __init__(self, tolerance=1e-8, epoch=0, learning_rate=0.01):
        self.coef_ = None
        self.intercept_ = None
        self.tolerance = tolerance
        self.epoch = epoch
        self.learningRate = learning_rate
        
    def predict(self,X):
        return  np.argmax(self.predict_proba(X), axis=1)
    
    def fit(self, X, y):
        k = len(np.unique(y))
        y_one_hot = self.one_hot(y.tolist(),k)
        m = X.shape[1]
        self.intercept_ = np.random.uniform(-0.01, 0.01, size=k)
        self.coef_ = np.random.uniform(-0.01, 0.01, size=(m,k))
        prev_cost = float('inf');
        cost = 1
        iteration = 0
        while iteration <= self.epoch if self.epoch else abs(prev_cost-cost) > self.tolerance:
            prev_cost = cost
            predicts = self.predict_proba(X)
            cost = self.cross_entropy_cost(predicts,y_one_hot)
            self.GradientDescent(predicts, y_one_hot, X)
            iteration += 1;
        return cost
    
    def one_hot(self, y, num_classes):
        return np.eye(num_classes)[y]
    
    def softmaxFunc(self,z):
        return np.exp(z) / np.sum(np.exp(z), axis = 1, keepdims = True)
    
    def predict_proba(self,X):
        z = np.dot(X, self.coef_) + self.intercept_    
        return self.softmaxFunc(z)
    
    def cross_entropy_cost(self,y_hat,y):
        return np.mean(-np.sum(y * np.log(y_hat), axis = 1))
    
    def GradientDescent(self,y_hat,y,X):
        m = y.shape[0]
        dw = (1/m) * np.dot(X.T, (y_hat - y))
        db = (1/m) * np.sum(y_hat - y)    
        self.coef_ = self.coef_ - self.learningRate * dw
        self.intercept_ = self.intercept_ - self.learningRate * db