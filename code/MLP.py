import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
class MLP:
    def __init__(
        self, input_size, hidden_layers, output_size, activations=None, 
        default_weights=None, default_biases=None, save_history=True
        ):
        """
        Initialize the MLP with configurable activations per layer.
        :param input_size: Number of input features
        :param hidden_layers: List of units in each hidden layer
        :param output_size: Number of output classes
        :param activations: List of activation functions for each layer
        """
        self.layers = [input_size] + hidden_layers + [output_size]
        self.weights = []
        self.biases = []

        if activations is None:
            activations = ['relu'] * len(hidden_layers) + ['softmax']
        self.activations = activations

        for i in range(len(self.layers) - 1):
            fan_in = self.layers[i]
            fan_out = self.layers[i + 1]
            activation = self.activations[i]

            if activation == 'tanh': # Xavier initialization
                limit = np.sqrt(6 / (fan_in + fan_out))
                w = np.random.uniform(-limit, limit, size=(fan_in, fan_out))
            elif activation in ['relu', 'leaky_relu']: # He initialization
                std = np.sqrt(2 / fan_in)
                w = np.random.randn(fan_in, fan_out) * std
            else:
                w = np.random.randn(fan_in, fan_out) * 0.01  # Default initialization

            self.weights.append(w)
            self.biases.append(np.zeros((1, fan_out)))
        
        # Save history of loss
        self.save_history = save_history
        self.history = {} #{epoch: loss}

        print("Model initialized with the following configuration:")
        print(f"Weights: min={np.min([np.min(w) for w in self.weights])}, max={np.max([np.max(w) for w in self.weights])}")

        self.hist_z_max = []
        self.hist_z_min = []

    # Activation functions and their derivatives
    def relu(self, z): return np.maximum(0, z)
    def relu_derivative(self, z, alpha=0.01): return np.where(z > 0, 1, alpha)

    def tanh(self, z): return np.tanh(z)
    def tanh_derivative(self, z): 
        print(f"---\nz in tanh_derivative: {z}, tanh(z) in tanh_derivative: {np.tanh(z)}, 1 - tanh(z) ** 2 in tanh_derivative: {1 - np.tanh(z) ** 2}")
        return 1 - np.tanh(z) ** 2
    def leaky_relu(self, z, alpha=0.01):
        return np.where(z > 0, z, alpha * z)

    def leaky_relu_derivative(self, z, alpha=0.01):
        dz = np.ones_like(z)
        dz[z < 0] = alpha
        return dz
    
    def softmax(self, z):
        if np.isnan(z).any() or np.isinf(z).any():
            print("NAN in z")
            print(z)
            plt.plot(self.hist_z_max)
            plt.plot(self.hist_z_min)
            plt.legend(["max", "min"])
            plt.yscale('log')
            plt.savefig("z_hist.png")
            raise ValueError("NAN in z")
        else :
            self.hist_z_max.append(np.max(z)) 
            self.hist_z_min.append(np.min(z))

        z_max = np.max(z, axis=1, keepdims=True)
        z_norm = z - z_max
        clip = 1e6
        z_norm = np.clip(z_norm, -clip, clip)
        exp_z = np.exp(z_norm)
        sum_exp_z = np.sum(exp_z, axis=1, keepdims=True) + 1e-8
        softmax = exp_z / sum_exp_z
        return softmax

    def get_activation(self, name):
        if name == 'relu': return self.relu
        elif name == 'tanh': return self.tanh
        elif name == 'leaky_relu': return self.leaky_relu
        elif name == 'softmax': return self.softmax
        else: raise ValueError(f"Unsupported activation: {name}")

    def get_activation_derivative(self, name):
        if name == 'relu': return self.relu_derivative
        elif name == 'leaky_relu': return self.leaky_relu_derivative
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
            z = np.dot(self.a[-1], self.weights[i]) + self.biases[i]
            self.z.append(z)

            # Apply layer-specific activation
            activation_func = self.get_activation(self.activations[i])
            self.a.append(activation_func(z))

        return self.a[-1]

    def backward(self, X, y, lr):
        m = X.shape[0]
        y_pred = self.a[-1]
        dz = y_pred - y  # Shape: (batch_size, num_classes)
        for i in reversed(range(len(self.weights))):
            # Compute gradients
            dw = np.dot(self.a[i].T, dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m

            # Clip gradients to prevent exploding gradients
            max_grad_norm = 100_000_000.0
            dw = np.clip(dw, -max_grad_norm, max_grad_norm)
            db = np.clip(db, -max_grad_norm, max_grad_norm)

            # Update weights and biases
            self.weights[i] -= lr * dw
            self.biases[i] -= lr * db
            # Backpropagate the error
            if i > 0:
                activation_derivative_func = self.get_activation_derivative(self.activations[i - 1])
                da = np.dot(dz, self.weights[i].T)
                derivative = activation_derivative_func(self.z[i - 1])
                dz = da * derivative

    def fit(self, X, y, epochs=50, lr=0.01, batch_size=32, save_weights=True, path_prefix=""):
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
            loss = round(-np.mean(np.sum(y * np.log(self.a[-1] + 1e-8), axis=1)), 4)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {loss}")
            if self.save_history:
                self.history[epoch] = loss
            
        # Save weights and biases
        if save_weights:
            if path_prefix:
                path_prefix = path_prefix.replace(' ','').replace('(','').replace(')','')
                directory = f"weights/{path_prefix}"
            else:
                directory = "weights"
            
            os.makedirs(directory, exist_ok=True)  # Create the directory recursively
            # Save weights and biases
            with open(f"{directory}/weights.pkl", "wb") as f:
                pickle.dump(self.weights, f)
            with open(f"{directory}/biases.pkl", "wb") as f:
                pickle.dump(self.biases, f)
        

    def predict(self, X):
        """
        Make predictions.
        :param X: Input data
        :return: Predicted class labels
        """
        probabilities = self.forward(X)
        return np.argmax(probabilities, axis=1)
