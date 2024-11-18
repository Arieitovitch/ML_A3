import numpy as np

class MLP:
    def __init__(self, input_size, hidden_layers, output_size, activations=None):
        """
        Initialize the MLP with configurable activations per layer.
        :param input_size: Number of input features
        :param hidden_layers: List of units in each hidden layer
        :param output_size: Number of output classes
        :param activations: List of activation functions for each layer
        """
        self.layers = [input_size] + hidden_layers + [output_size]
        self.weights = [np.random.randn(self.layers[i], self.layers[i + 1]) * 0.01 
                        for i in range(len(self.layers) - 1)]
        self.biases = [np.zeros((1, self.layers[i + 1])) for i in range(len(self.layers) - 1)]
        
        # Default activations if none provided
        if activations is None:
            activations = ['relu'] * (len(hidden_layers)) + ['softmax']
        self.activations = activations

    # Activation functions and their derivatives
    def relu(self, z): return np.maximum(0, z)
    def relu_derivative(self, z): return (z > 0).astype(float)

    def tanh(self, z): return np.tanh(z)
    def tanh_derivative(self, z): return 1 - np.tanh(z) ** 2

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def get_activation(self, name):
        if name == 'relu': return self.relu
        elif name == 'tanh': return self.tanh
        elif name == 'softmax': return self.softmax
        else: raise ValueError(f"Unsupported activation: {name}")

    def get_activation_derivative(self, name):
        if name == 'relu': return self.relu_derivative
        elif name == 'tanh': return self.tanh_derivative
        else: raise ValueError(f"Unsupported derivative for activation: {name}")

    def forward(self, X):
        """
        Perform forward propagation.
        :param X: Input data
        :return: Output predictions
        """
        self.a = [X]
        self.z = []

        for i in range(len(self.weights)):
            z = self.a[-1] @ self.weights[i] + self.biases[i]
            self.z.append(z)

            # Apply layer-specific activation
            activation_func = self.get_activation(self.activations[i])
            self.a.append(activation_func(z))

        return self.a[-1]

    def backward(self, X, y, lr):
        """
        Perform backward propagation.
        :param X: Input data
        :param y: One-hot encoded labels
        :param lr: Learning rate
        """
        m = X.shape[0]
        y_pred = self.a[-1]
        dz = y_pred - y  # Derivative of loss w.r.t. output

        for i in reversed(range(len(self.weights))):
            dw = self.a[i].T @ dz / m
            db = np.sum(dz, axis=0, keepdims=True) / m

            if i > 0:
                activation_derivative_func = self.get_activation_derivative(self.activations[i - 1])
                dz = dz @ self.weights[i].T * activation_derivative_func(self.z[i - 1])

            # Update weights and biases
            self.weights[i] -= lr * dw
            self.biases[i] -= lr * db

    def fit(self, X, y, epochs=100, lr=0.01, batch_size=64):
        """
        Train the model using mini-batch gradient descent.
        :param X: Training data
        :param y: One-hot encoded labels
        :param epochs: Number of training epochs
        :param lr: Learning rate
        :param batch_size: Batch size
        """
        for epoch in range(epochs):
            # Shuffle the data
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]

            # Mini-batch gradient descent
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                self.forward(X_batch)
                self.backward(X_batch, y_batch, lr)

            # Print loss for debugging (optional)
            if epoch % 10 == 0:
                loss = -np.mean(np.sum(y * np.log(self.a[-1] + 1e-8), axis=1))
                print(f"Epoch {epoch}: Loss = {loss:.4f}")

    def predict(self, X):
        """
        Make predictions.
        :param X: Input data
        :return: Predicted class labels
        """
        probabilities = self.forward(X)
        return np.argmax(probabilities, axis=1)
