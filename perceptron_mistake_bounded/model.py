# perceptron_mistake_bounded/model.py

import numpy as np

class PerceptronMistakeBounded:
    def __init__(self, learning_rate=1.0, max_epochs=1000):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weights = None
        self.bias = 0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.max_epochs):
            mistakes = 0
            for x_i, y_i in zip(X, y):
                linear_output = np.dot(x_i, self.weights) + self.bias
                predicted = np.sign(linear_output)

                if predicted != y_i:
                    self.weights += self.learning_rate * y_i * x_i
                    self.bias += self.learning_rate * y_i
                    mistakes += 1

            print(f"Epoch {epoch+1}: {mistakes} mistakes")
            if mistakes == 0:
                break

    def predict(self, X):
        return np.sign(np.dot(X, self.weights) + self.bias)
