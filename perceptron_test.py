from nn.perceptron import Perceptron
import numpy as np

# Create a perceptron
perceptron = Perceptron()

# Train the perceptron
# AND
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([-1, -1, -1, 1])

perceptron.train(X, y)

# Test the perceptron
for i in range (X.shape[0]):
    print(perceptron.predict(X[i]))
