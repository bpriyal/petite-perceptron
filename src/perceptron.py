import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=100):
        """
        Initializes the Perceptron parameters.
        Args:
            learning_rate (float): Step size for weight updates (between 0 and 1).
            epochs (int): Number of times to iterate over the dataset.
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def _step_function(self, x):
        """
        Activation function: Returns 1 if input >= 0, else 0.
        Source: [5], [4]
        """
        return 1 if x >= 0 else 0

    def fit(self, X, y):
        """
        Trains the perceptron on the provided data.
        Args:
            X (array-like): Input feature vectors.
            y (array-like): Target values (labels).
        """
        n_samples, n_features = X.shape
        
        # Initialize weights as zeros (or small random numbers)
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Training Loop
        for epoch in range(self.epochs):
            for i, x_i in enumerate(X):
                # 1. Forward Pass: Calculate weighted sum + bias
                linear_output = np.dot(x_i, self.weights) + self.bias
                
                # 2. Activation: Apply step function to get prediction
                y_predicted = self._step_function(linear_output)
                
                # 3. Calculate Error: Target - Prediction
                # Source: [6], [4]
                error = y[i] - y_predicted
                
                # 4. Backward Pass: Update Weights and Bias only if there is error
                # Weight Update Rule: W = W + (learning_rate * error * input)
                # Bias Update Rule: B = B + (learning_rate * error)
                if error != 0:
                    update = self.learning_rate * error
                    self.weights += update * x_i
                    self.bias += update

    def predict(self, X):
        """
        Predicts labels for new input data.
        Source: [7], [4]
        """
        predictions = []
        for x_i in X:
            linear_output = np.dot(x_i, self.weights) + self.bias
            y_predicted = self._step_function(linear_output)
            predictions.append(y_predicted)
        return np.array(predictions)