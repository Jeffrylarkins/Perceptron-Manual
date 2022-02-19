import numpy as np


class perceptron:
    def __init__(self, eta, epochs):
        self.weights = np.random.randn(3)*1e-4  #Smaller vAlue
        print(f"initial weights before training:\n{self.weights}")
        self.eta = eta    #Learning Rate
        self.epochs = epochs
    
    def activationfunc(self, inputs, weights):
        z = np.dot(inputs, weights) #z = Wi*Xi
        return np.where(z > 0, 1,0)
    
    def fit(self, X,y):
        self.X = X
        self.y = y
        X_with_bias = np.c_[self.X, -np.ones((len(self.X),1))]##concatenation
        print(f"X with bias : \n{X_with_bias}")
        
        for epoch in range(self.epochs): ##ITERATION
            print("--"*10)
            print(f"for epoch :{epoch}")
            print("--"*10)
              ##FORWARD PROPOGATION    
            y_hat = self.activationfunc(X_with_bias, self.weights) #X_with bias must be multiplied by Weights Z= X_withbias*Weights
            print(f"predicted value after forward pass:\n{y_hat}")   # done in def activationfunc
            self.error = self.y - y_hat 
            print(f"the errors are:\n{self.error}")
                  
            self.weights = self.weights + self.eta*np.dot(X_with_bias.T, self.error)##BACKWARD PROPOGATION
            print(f"updated weights after epoch:{epoch}/{self.epochs}: {self.weights}")
            print('######'*10)
    def predict(self, X):
        X_with_bias = np.c_[X, -np.ones((len(X),1))]
        return self.activationfunc(X_with_bias, self.weights)
    def total_loss(self):
        total_loss = np.sum(self.error)
        print(f"Total loss : {total_loss}")
        return total_loss