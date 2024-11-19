import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import copy

class MLPREG:
    def __init__(self, input_size, hidden_layers, output_size, activations=None,
                 default_weights=None, default_biases=None, save_history=True):
        self.layers = [input_size] + hidden_layers + [output_size]
        self.weights = []
        self.biases = []

        if activations is None:
            activations = ['relu'] * len(hidden_layers) + ['softmax']
        self.activations = activations

        if not default_biases or not default_weights:
            for i in range(len(self.layers) - 1):
                fan_in = self.layers[i]
                fan_out = self.layers[i + 1]
                activation = self.activations[i]

                if activation == 'tanh':  # Xavier initialization
                    limit = np.sqrt(6 / (fan_in + fan_out))
                    w = np.random.uniform(-limit, limit, size=(fan_in, fan_out))
                elif activation in ['relu', 'leaky_relu']:  # He initialization
                    std = np.sqrt(2 / fan_in)
                    w = np.random.randn(fan_in, fan_out) * std
                else:
                    w = np.random.randn(fan_in, fan_out) * 0.01  # Default initialization

                self.weights.append(w)
                self.biases.append(np.zeros((1, fan_out)))
        else:
            self.weights = default_weights
            self.biases = default_biases

        self.save_history = save_history
        self.history = {}
        self.val_history = {}  # For validation loss history
        self.historic_weights = {} # {epoch: [weights, biases]}

        print("Model initialized with the following configuration:")
        print(f"Weights: min={np.min([np.min(w) for w in self.weights])}, "
              f"max={np.max([np.max(w) for w in self.weights])}")

    # Activation functions and their derivatives
    def relu(self, z): return np.maximum(0, z)
    def relu_derivative(self, z): return np.where(z > 0, 1, 0)

    def tanh(self, z): return np.tanh(z)
    def tanh_derivative(self, z): return 1 - np.tanh(z) ** 2

    def leaky_relu(self, z, alpha=0.01):
        return np.where(z > 0, z, alpha * z)
    def leaky_relu_derivative(self, z, alpha=0.01):
        dz = np.ones_like(z)
        dz[z < 0] = alpha
        return dz

    def softmax(self, z):
        z_max = np.max(z, axis=1, keepdims=True)
        z_norm = z - z_max
        clip = 1e9
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
        self.a = [X]
        self.z = []

        for i in range(len(self.weights)):
            z = np.dot(self.a[-1], self.weights[i]) + self.biases[i]
            self.z.append(z)

            if i == len(self.weights) - 1:
                # Do not apply activation to the output layer to get logits
                self.a.append(z)  # Logits
            else:
                activation_func = self.get_activation(self.activations[i])
                self.a.append(activation_func(z))

        self.logits = self.a[-1]
        return self.logits  # Return logits

    def compute_loss(self, y_true, logits):
        logits_stable = logits - np.max(logits, axis=1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(logits_stable), axis=1, keepdims=True))
        logits_true_class = np.sum(logits_stable * y_true, axis=1, keepdims=True)
        loss = -logits_true_class + log_sum_exp
        loss = np.mean(loss)
        return loss

    def backward(self, X, y, lr, l1_lambda=0.0, l2_lambda=0.0):
        m = X.shape[0]
        logits = self.logits
        y_pred = self.softmax(logits)  # Apply softmax to logits
        dz = y_pred - y

        for i in reversed(range(len(self.weights))):
            dw = np.dot(self.a[i].T, dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m

            # Add regularization terms
            if l1_lambda > 0:
                dw += (l1_lambda / m) * np.sign(self.weights[i])
            if l2_lambda > 0:
                dw += (l2_lambda / m) * self.weights[i]

            # Clip gradients
            max_grad_norm = 1e8
            dw = np.clip(dw, -max_grad_norm, max_grad_norm)
            db = np.clip(db, -max_grad_norm, max_grad_norm)

            # Update weights and biases
            self.weights[i] -= lr * dw
            self.biases[i] -= lr * db

            if i > 0:
                activation_derivative_func = self.get_activation_derivative(self.activations[i - 1])
                da = np.dot(dz, self.weights[i].T)
                derivative = activation_derivative_func(self.z[i - 1])
                dz = da * derivative

    def fit(self, X, y, epochs=100, lr=0.01, batch_size=64, l1_lambda=0.0, l2_lambda=0.0,
            save_weights=True, path_prefix="", X_val=None, y_val=None, patience=5, early_stopping = True):
        for epoch in range(epochs):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]

            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                logits = self.forward(X_batch)
                self.backward(X_batch, y_batch, lr, l1_lambda, l2_lambda)

            # Compute training loss
            logits = self.forward(X)
            loss = self.compute_loss(y, logits)
            # Add regularization terms
            l1_loss = l1_lambda * np.sum([np.sum(np.abs(w)) for w in self.weights]) / X.shape[0]
            l2_loss = l2_lambda * 0.5 * np.sum([np.sum(w ** 2) for w in self.weights]) / X.shape[0]
            total_loss = loss + l1_loss + l2_loss
            total_loss = round(total_loss, 4)

            # Compute validation loss if validation data is provided
            # Regularization terms are not included in the validation loss to assess how well the model generalizes, independent of the regularization penalties applied during training.
            if X_val is not None and y_val is not None:
                val_logits = self.forward(X_val)
                val_loss = self.compute_loss(y_val, val_logits)
                val_total_loss = round(val_loss, 4)  # Exclude regularization terms

                # Validation accuracy -> possible future extension
                # val_predictions = self.predict(X_val)
                # val_true_labels = np.argmax(y_val, axis=1)
                # val_accuracy = np.mean(val_predictions == val_true_labels)
                # self.val_accuracy_history[epoch] = val_accuracy
            else:
                val_total_loss = None

            # Save losses
            if self.save_history:
                self.history[epoch] = total_loss
                if val_total_loss is not None:
                    self.val_history[epoch] = val_total_loss
                self.historic_weights[epoch] = copy.deepcopy([self.weights, self.biases])

            # Print losses for monitoring
            if epoch % 10 == 0:
                if val_total_loss is not None:
                    print(f"Epoch {epoch}: Training Loss = {total_loss}, Validation Loss = {val_total_loss}")
                else:
                    print(f"Epoch {epoch}: Training Loss = {total_loss}")

            # Early Stopping -> possible future extension
            # if early_stopping and val_total_loss is not None:
            #     if epoch > 0 and val_total_loss > self.val_history[epoch - 1]:
            #         early_stop_counter += 1
            #         if early_stop_counter >= patience:
            #             print("Early stopping triggered.")
            #             break
            #     else:
            #         early_stop_counter = 0


        if save_weights:
            if path_prefix:
                path_prefix = path_prefix.replace(' ', '').replace('(', '').replace(')', '')
                directory = f"weights/{path_prefix}"
            else:
                directory = "weights"

            os.makedirs(directory, exist_ok=True)
            with open(f"{directory}/weights.pkl", "wb") as f:
                pickle.dump(self.weights, f)
            with open(f"{directory}/biases.pkl", "wb") as f:
                pickle.dump(self.biases, f)
            with open(f"{directory}/historic_weights.pkl", "wb") as f:
                pickle.dump(self.historic_weights, f)

    def predict(self, X):
        probabilities = self.forward(X)
        return np.argmax(probabilities, axis=1)
