import numpy as np

# z = wx_1 + wx_2 + ... wx_n + w0 
# prediction  = 1/1(1 + e^-z)
class Model():
    def __init__(self, learning_rate=0.01,epoch=0):
        self.intercept_ = None
        self.coef_ = None
        self.epoch = epoch
        self.learningRate = learning_rate

    def GradientDescent(self,y_hat,y,X):
        m = y.shape[0]
        dw = (1/m) * np.dot(X.T, (y_hat - y))
        db = (1/m) * np.sum(y_hat - y)    
        self.coef_ = self.coef_ - self.learningRate * dw
        self.intercept_ = self.intercept_ - self.learningRate * db

    def predict(self,X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def predict_proba(self,X):
        z = np.dot(X, self.coef_) + self.intercept_        
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        self.intercept_ = [0] 
        self.coef_ = np.array([0] * X.shape[1])
        tol = 1e-8
        prev_cost = float('inf');
        cost = 1
        iteration = 0
        while iteration <= self.epoch if self.epoch else abs(prev_cost-cost) > tol:
            prev_cost = cost
            predicts = self.predict_proba(X)
            cost = self.compute_loss(predicts,y)
            self.GradientDescent(predicts, y, X)
            iteration += 1;
        return cost
        
    def compute_loss(self,y_hat, y):
        m = y.shape[0]  
        epsilon = 1e-9 
        y_hat_clipped = np.clip(y_hat, epsilon, 1 - epsilon) #to avoid log(0)        
        loss = - (1/m) * np.sum(y * np.log(y_hat_clipped) + (1 - y) * np.log(1 - y_hat_clipped))
        return loss