import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import copy
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

        if not default_weights or not default_biases:
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
        else:
            self.weights = default_weights
            self.biases = default_biases
        
        # Save history of loss
        self.save_history = save_history
        self.history = {}      # {epoch: loss}
        self.val_history = {}  # {epoch: val_loss}
        self.historic_weights = {} # {epoch: [weights, biases]}
        self.accuracy = {}     # {epoch: accuracy}

        print("Model initialized with the following configuration:")
        print(f"Weights: min={np.min([np.min(w) for w in self.weights])}, max={np.max([np.max(w) for w in self.weights])}")
        self.hist_z_max = []
        self.hist_z_min = []

    # Activation functions and their derivatives
    def relu(self, z): return np.maximum(0, z)
    def relu_derivative(self, z, alpha=0.01): return np.where(z > 0, 1, alpha)

    def tanh(self, z):
        if np.isnan(z).any() or np.isinf(z).any():
            raise ValueError("NaN or Inf detected in tanh inputs")
        output = np.tanh(z)
        if np.isnan(output).any() or np.isinf(output).any():
            raise ValueError("NaN or Inf detected in tanh outputs")
        return output

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
        z = np.clip(z, -50, 50) 
        z_max = np.max(z, axis=1, keepdims=True)
        z_norm = z - z_max
        exp_z = np.exp(z_norm)
        sum_exp_z = np.sum(exp_z, axis=1, keepdims=True) + 1e-8  
        softmax = exp_z / sum_exp_z
        softmax = np.clip(softmax, 1e-8, 1 - 1e-8)
        
        if np.isnan(softmax).any() or np.isinf(softmax).any():
            raise ValueError("NaN or Inf detected in softmax output")
        
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
            if np.isnan(z).any() or np.isinf(z).any():
                raise ValueError(f"NaN or Inf detected in logits (z) at layer {i}")
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
            
            if np.isnan(dw).any() or np.isnan(db).any():
                raise ValueError("NaN detected in backward pass (gradients)")


            # Clip gradients to prevent exploding gradients
            max_grad_norm = 100.0
            dw = np.clip(dw, -max_grad_norm, max_grad_norm)
            db = np.clip(db, -max_grad_norm, max_grad_norm)

            # Update weights and biases
            self.weights[i] -= lr * dw
            self.biases[i] -= lr * db
            
            for w in self.weights:
                if np.isnan(w).any() or np.isinf(w).any():
                    raise ValueError("NaN or Inf detected in weights")

            for b in self.biases:
                if np.isnan(b).any() or np.isinf(b).any():
                    raise ValueError("NaN or Inf detected in biases")

            
            # Backpropagate the error
            if i > 0:
                activation_derivative_func = self.get_activation_derivative(self.activations[i - 1])
                da = np.dot(dz, self.weights[i].T)
                derivative = activation_derivative_func(self.z[i - 1])
                dz = da * derivative

    def fit(self, X, y, epochs=50, lr=0.01, batch_size=32, save_weights=True, path_prefix="", 
            y_val=None, X_val=None,
            y_test=None, X_test=None
        ):
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

            # Compute training loss
            logits = self.forward(X)
            loss = self.compute_loss(y, logits)
            
            # Compute validation loss if validation data is provided
            if X_val is not None and y_val is not None:
                val_logits = self.forward(X_val)
                val_loss = self.compute_loss(y_val, val_logits)
            else:
                val_loss = None

            # Save losses
            
            log_probs = np.log(self.a[-1] + 1e-8)
            if np.isnan(log_probs).any():
                raise ValueError("NaN detected in log probabilities during loss calculation")

            loss = -np.mean(np.sum(y * log_probs, axis=1))
            if np.isnan(loss):
                raise ValueError("NaN detected in loss value")

            
            loss = round(-np.mean(np.sum(y * np.log(self.a[-1] + 1e-8), axis=1)), 4)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {loss}")
            if self.save_history:
                self.history[epoch] = loss
                if val_loss is not None:
                    self.val_history[epoch] = val_loss
                self.historic_weights[epoch] = copy.deepcopy([self.weights, self.biases])
            
            if X_test is not None and y_test is not None:
                # Compute accuracy
                y_preds = self.predict(X_test)
                y_true = np.argmax(y_test, axis=1)
                accuracy = np.mean(y_preds == y_true)
                self.accuracy[epoch] = accuracy

            # Print losses for monitoring
            if epoch % 10 == 0:
                if val_loss is not None:
                    print(f"Epoch {epoch}: Training Loss = {loss:.4f}, Validation Loss = {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch}: Training Loss = {loss:.4f}")

            # Early stopping -> possible future extension
            # if val_total_loss is not None:
            #     if epoch > 0 and val_total_loss > self.val_history[epoch - 1]:
            #         early_stop_counter += 1
            #         if early_stop_counter >= patience:
            #             print("Early stopping triggered.")
            #             break
            #     else:
            #         early_stop_counter = 0

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
            # Save history of weights
            with open(f"{directory}/historic_weights.pkl", "wb") as f:
                pickle.dump(self.historic_weights, f)
            

    def predict(self, X):
        """
        Make predictions.
        :param X: Input data
        :return: Predicted class labels
        """
        probabilities = self.forward(X)
        return np.argmax(probabilities, axis=1)
    
    def compute_loss(self, y_true, logits):
        logits_stable = logits - np.max(logits, axis=1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(logits_stable), axis=1, keepdims=True))
        logits_true_class = np.sum(logits_stable * y_true, axis=1, keepdims=True)
        loss = -logits_true_class + log_sum_exp
        loss = np.mean(loss)
        return loss

