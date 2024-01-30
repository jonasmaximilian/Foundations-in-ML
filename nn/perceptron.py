import numpy as np 

# maps input (real-valued vector) to output value -1 or 1
class Perceptron:
    def __init__(self, epochs=100):
        self.weights = None
        self.bias = None
        self.epochs = epochs
    
    def train(self, X, y):
        # Initialize the weights and bias
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for epoch in range(self.epochs):
            for i in range (X.shape[0]):
                y_hat= self.predict(X[i])
                error = y[i] - y_hat
                self.weights += 1/2 * error * X[i]
                self.bias += 1/2 * error 
    def predict(self, x):
        # signum of weighted sum
        activation = np.dot(self.weights, x) + self.bias
        return np.sign(activation)

