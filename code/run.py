# this file is going to be run by slurm
import time
import data_aquire
from MLP import MLP
from MLPREG import MLPREG
import numpy as np
import os
from medmnist import OrganAMNIST# type: ignore
from medmnist import INFO# type: ignore
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms # type: ignore
from PIL import Image
from CNN import CNN, CNN128
from torchvision import models # type: ignore
import matplotlib.pyplot as plt
import pickle
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_histories(histories, filename, directory = "histories"):
    os.makedirs(directory, exist_ok=True)
    with open(f"{directory}/{filename}.pkl", "wb") as pkl_file:
        pickle.dump(histories, pkl_file)

def prepare_128px_data():
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    train_dataset = OrganAMNIST(split='train', download=True, transform=transform)
    test_dataset = OrganAMNIST(split='test', download=True, transform=transform)

    X_train_128 = []
    y_train_128 = []
    for img, label in train_dataset:
        X_train_128.append(img.numpy().reshape(-1)) 
        y_train_128.append(label.item())
    X_train_128 = np.array(X_train_128)
    y_train_128 = np.array(y_train_128)

    X_test_128 = []
    y_test_128 = []
    for img, label in test_dataset:
        X_test_128.append(img.numpy().reshape(-1))
        y_test_128.append(label.item())
    X_test_128 = np.array(X_test_128)
    y_test_128 = np.array(y_test_128)

    return X_train_128, y_train_128, X_test_128, y_test_128

# Load the preprocessed data
def load_data(file_path):
    data = np.load(file_path)
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    return X_train, y_train, X_val, y_val, X_test, y_test

def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y.astype(int)]

def load_unnormalized_data():
    data = np.load(os.path.join("code/code/processed_data", "mlp_data_unnormalized.npz"))
    X_train = data['X_train'] 
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    return X_train, y_train, X_test, y_test

data_path = os.path.join("code/code/processed_data", "mlp_data.npz")
print(data_path)
X_train, y_train, X_val, y_val, X_test, y_test = load_data(data_path)
input_size = X_train.shape[1]
hidden_layers = []
# Determine the number of classes
num_classes = len(np.unique(y_train))

# One-hot encode the labels
y_train_one_hot = one_hot_encode(y_train, num_classes)
y_test_one_hot = one_hot_encode(y_test, num_classes)
y_val_one_hot = one_hot_encode(y_val, num_classes)

# Import the MLP class (assuming it's defined in mlp.py)
# from mlp import MLP
X_train_un, y_train_un, X_test_un, y_test_un = load_unnormalized_data()
y_train_un_one_hot = one_hot_encode(y_train_un, num_classes)

transform = transforms.Compose([
    transforms.ToTensor(),  # Converts images to PyTorch tensors and scales pixel values to [0, 1]
])

train_dataset = OrganAMNIST(split='train', download=True, transform=transform)
test_dataset = OrganAMNIST(split='test', download=True, transform=transform)

# Create DataLoaders
batch_size = 64
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    
        
# Evaluate the model
def evaluate_cnn(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            # Move tensors to the same device as the model
            images, labels = images.to(device), labels.to(device)
            
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            labels = labels.squeeze().long()
            
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if total == 0:
        return 0.0
    
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return round(accuracy, 4)




def train_transfer_model(model, train_loader, criterion, optimizer, epochs=10, save_weights=True, path_prefix=""):
    history = {}
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = adjust_labels(labels).to(device)
            # Zero gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(images)
            # Compute loss
            loss = criterion(outputs, labels)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")
        history[epoch] = round(total_loss,4)
        
    if save_weights:
        if path_prefix:
            path_prefix = path_prefix.replace(' ','').replace('(','').replace(')','')
            directory = f"weights/{path_prefix}"
        else:
            directory = "weights"
        os.makedirs(directory, exist_ok=True)  # Create the directory recursively
        # Save weights and biases
        torch.save(model.state_dict(), f"{directory}/weights.pth")
    return history
        
            
def evaluate_transfer_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = adjust_labels(labels).to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return accuracy


# Function to train and evaluate a model
def train_and_evaluate(model, model_name, epochs=100, lr = 0.01, batch_size=64):
    print(f"Training {model_name}")
    model.fit(
        X_train, y_train_one_hot, 
        epochs=epochs, lr=lr, batch_size=batch_size, 
        save_weights=True, path_prefix=model_name, 
        y_val = y_val_one_hot, X_val = X_val,
        y_test = y_test_one_hot, X_test = X_test
    )
    y_pred = model.predict(X_test)
    test_accuracy = np.mean(y_pred == y_test)
    print(f"{model_name} Test Accuracy: {test_accuracy * 100:.2f}%\n")
    return test_accuracy


def task1():
    # Model 1: No hidden layers
    activations = ['softmax']  # Only output activation
    model1 = MLP(input_size, [], num_classes, activations)
    acc1 = train_and_evaluate(model1, "Model 1 (No Hidden Layers)", epochs=100, lr = 0.001, batch_size=128)

    # Model 2: One hidden layer
    hidden_layers = [256]
    activations = ['relu', 'softmax']
    model2 = MLP(input_size, hidden_layers, num_classes, activations)
    acc2 = train_and_evaluate(model2, "Model 2 (One Hidden Layer)", epochs=100, lr = 0.001, batch_size=128)

    # Model 3: Two hidden layers
    hidden_layers = [256, 256]
    activations = ['relu', 'relu', 'softmax']
    model3 = MLP(input_size, hidden_layers, num_classes, activations)
    acc3 = train_and_evaluate(model3, "Model 3 (Two Hidden Layers)", epochs=100, lr = 0.001, batch_size=128)
    
    # Write histories to memory
    histories = {
        "Model 1 (No Hidden Layers)": model1.history,
        "Model 2 (One Hidden Layer)": model2.history,
        "Model 3 (Two Hidden Layers)": model3.history,
    }
    save_histories(histories, "loss_histories_task1")
    val_losses = {
        "Model 1 (No Hidden Layers)": model1.val_history,
        "Model 2 (One Hidden Layer)": model2.val_history,
        "Model 3 (Two Hidden Layers)": model3.val_history,
    }
    save_histories(val_losses, "val_loss_histories_task1")
    acc_histories = {
        "Model 1 (No Hidden Layers)": model1.accuracy,
        "Model 2 (One Hidden Layer)": model2.accuracy,
        "Model 3 (Two Hidden Layers)": model3.accuracy,
    }
    save_histories(acc_histories, "accuracy_histories_task1")

    # Return the test accuracies
    return {
        "Model 1 (No Hidden Layers)": acc1, 
        "Model 2 (One Hidden Layer)": acc2, 
        "Model 3 (Two Hidden Layers)": acc3,
    }
    
    
def task2():
    # Model with tanh activations
    activations_tanh = ['tanh', 'tanh', 'softmax']
    model_tanh = MLP(input_size, hidden_layers, num_classes, activations_tanh)
    acc_tanh = train_and_evaluate(model_tanh, "Model with Tanh Activations", epochs=100, lr = 0.01, batch_size=128)

    # Model with Leaky ReLU activations
    activations_leaky_relu = ['leaky_relu', 'leaky_relu', 'softmax']
    model_leaky_relu = MLP(input_size, hidden_layers, num_classes, activations_leaky_relu)
    acc_leaky_relu = train_and_evaluate(model_leaky_relu, "Model with Leaky ReLU Activations", epochs=100, lr = 0.01, batch_size=128)
    
    # Write histories to memory
    histories = {
        "Model with Tanh Activations": model_tanh.history,
        "Model with Leaky ReLU Activations": model_leaky_relu.history,
    }
    save_histories(histories, "loss_histories_task2")
    val_losses = {
        "Model with Tanh Activations": model_tanh.val_history,
        "Model with Leaky ReLU Activations": model_leaky_relu.val_history,
    }
    save_histories(val_losses, "val_loss_histories_task2")
    acc_histories = {
        "Model with Tanh Activations": model_tanh.accuracy,
        "Model with Leaky ReLU Activations": model_leaky_relu.accuracy,
    }
    save_histories(acc_histories, "accuracy_histories_task2")
    
    return {
        "Model with Tanh Activations": acc_tanh,
        "Model with Leaky ReLU Activations": acc_leaky_relu,
    }
    
def task3():
    activations = ['leaky_relu', 'leaky_relu', 'softmax']
    model_l1 = MLPREG(input_size, hidden_layers, num_classes, activations)
    print("Training Model with L1 Regularization")
    model_l1.fit(
        X_train, y_train_one_hot, 
        epochs=100, lr=0.005, batch_size=64, l1_lambda=0.001, 
        save_weights=True, path_prefix="task3_l1", 
        X_val=X_val, y_val=y_val_one_hot,
        y_test = y_test_one_hot, X_test = X_test
    )
    y_pred_l1 = model_l1.predict(X_test)
    acc_l1 = np.mean(y_pred_l1 == y_test)
    print(f"Model with L1 Regularization Test Accuracy: {acc_l1 * 100:.2f}%\n")
    
    model_l2 = MLPREG(input_size, hidden_layers, num_classes, activations)
    print("Training Model with L2 Regularization")
    
    model_l2.fit(
        X_train, y_train_one_hot, 
        epochs=100, lr=0.005, batch_size=64, l2_lambda=0.001, 
        save_weights=True, path_prefix="task3_l2", 
        X_val=X_val, y_val=y_val_one_hot,
        y_test = y_test_one_hot, X_test = X_test
    )
    y_pred_l2 = model_l2.predict(X_test)
    acc_l2 = np.mean(y_pred_l2 == y_test)
    print(f"Model with L2 Regularization Test Accuracy: {acc_l2 * 100:.2f}%\n")
    
    # Write histories to memory
    histories = {
        "Model with L1 Regularization": model_l1.history,
        "Model with L2 Regularization": model_l2.history,
    }
    save_histories(histories, "loss_histories_task3")
    val_losses = {
        "Model with L1 Regularization": model_l1.val_history,
        "Model with L2 Regularization": model_l2.val_history,
    }
    save_histories(val_losses, "val_loss_histories_task3")
    acc_histories = {
        "Model with L1 Regularization": model_l1.accuracy,
        "Model with L2 Regularization": model_l2.accuracy,
    }
    save_histories(acc_histories, "accuracy_histories_task3")
    
    return {
        "Model with L1 Regularization": acc_l1,
        "Model with L2 Regularization": acc_l2,
    }
    
def task4():
    activations = ['leaky_relu', 'leaky_relu', 'softmax']
    model_un = MLP(input_size, hidden_layers, num_classes, activations)

    # Train the model on unnormalized data
    print("Training Model on Unnormalized Data")
    model_un.fit(
        X_train_un, y_train_un_one_hot, 
        epochs=50, lr=0.01, batch_size=64, 
        save_weights=True, path_prefix="task4_un", 
        X_val=X_val, y_val=y_val_one_hot,
        y_test = y_test_one_hot, X_test = X_test_un
    )
    y_pred_un = model_un.predict(X_test_un)
    acc_un = np.mean(y_pred_un == y_test_un)
    print(f"Model on Unnormalized Data Test Accuracy: {acc_un * 100:.2f}%\n")
    
    # Write histories to memory
    histories = {"Model on Unnormalized Data": model_un.history}
    save_histories(histories, "loss_histories_task4")
    val_losses = {"Model on Unnormalized Data": model_un.val_history}
    save_histories(val_losses, "val_loss_histories_task4")
    acc_histories = {"Model on Unnormalized Data": model_un.accuracy}
    save_histories(acc_histories, "accuracy_histories_task4")

    return {
        "Model on Unnormalized Data": acc_un,
    }
    
def task5(): 
    X_train_128, y_train_128, X_test_128, y_test_128 = prepare_128px_data()
    
    # One-hot encode the labels
    num_classes = len(np.unique(y_train_128))
    y_train_128_one_hot = one_hot_encode(y_train_128, num_classes)
    y_test_128_one_hot = one_hot_encode(y_test_128, num_classes)
    
    # Adjust the input size
    input_size_128 = X_train_128.shape[1]
    hidden_layers = [256, 256]
    activations = ['relu', 'relu', 'softmax']   
    # Initialize the model
    model_128 = MLP(input_size_128, hidden_layers, num_classes, activations)
    
    # Train the model
    print("Training MLP on 128-Pixel Images")
    model_128.fit(
        X_train_128, y_train_128_one_hot,
        epochs=100, lr=0.01, batch_size=64,
        save_weights=True, path_prefix="task5_128",
        y_val=y_test_128_one_hot, X_val=X_test_128,
        y_test = y_test_128_one_hot, X_test = X_test_128
    )
    
    # Evaluate the model
    y_pred_128 = model_128.predict(X_test_128)
    acc_128 = np.mean(y_pred_128 == y_test_128)
    print(f"MLP on 128-Pixel Images Test Accuracy: {acc_128 * 100:.2f}%\n")
    
    # Write histories to memory
    histories = {
        "MLP on 128-Pixel Images": model_128.history,
    }
    save_histories(histories, "loss_histories_task5")
    val_losses = {
        "MLP on 128-Pixel Images": model_128.val_history,
    }
    save_histories(val_losses, "val_loss_histories_task5")
    acc_histories = {
        "MLP on 128-Pixel Images": model_128.accuracy,
    }
    save_histories(acc_histories, "accuracy_histories_task5")
    
    return {
        "MLP on 128-Pixel Images": acc_128,
    }


import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
from medmnist import OrganAMNIST

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def adjust_labels(labels):
    labels = labels.squeeze()
    if labels.dim() == 0:
        labels = labels.unsqueeze(0)
    return labels.long()


def save_metrics_plot(metrics, title, ylabel, save_path):
    """
    Save training metrics as a plot to the specified path.
    """
    epochs = range(1, len(metrics['loss']) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics['loss'], label='Loss', marker='o')
    plt.plot(epochs, metrics['accuracy'], label='Accuracy (%)', marker='o')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def train_cnn(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    metrics = {'loss': [], 'accuracy': []}

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            labels = labels.view(-1).long()
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        metrics['loss'].append(avg_loss)
        metrics['accuracy'].append(accuracy)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    return metrics


def evaluate_cnn(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            labels = labels.view(-1).long()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if total == 0:
        raise ValueError("Total test samples are zero. Check your test_loader.")

    accuracy = correct / total
    print(f"Correct: {correct}, Total: {total}, Accuracy: {accuracy * 100:.2f}%")
    return round(accuracy * 100, 2)


class TransferLearningModel(nn.Module):
    def __init__(self, num_classes):
        super(TransferLearningModel, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)


def task6():
    train_dataset = OrganAMNIST(split='train', download=True, transform=transforms.ToTensor())
    test_dataset = OrganAMNIST(split='test', download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    num_classes = len(np.unique(train_dataset.labels))
    cnn_model = CNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

    print("Training Task 6 CNN Model")
    start_time = time.time()
    metrics = train_cnn(cnn_model, train_loader, criterion, optimizer, epochs=10)
    end_time = time.time()

    training_time = end_time - start_time
    print(f"Task 6: Model training completed in {training_time:.2f} seconds.")

    save_metrics_plot(metrics, "Task 6: CNN Training Metrics", "Value", "plots/task6_metrics.png")

    accuracy = evaluate_cnn(cnn_model, test_loader)
    return {"Accuracy": accuracy, "Training Time (seconds)": training_time}


def task7():
    transform_128 = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    train_dataset_128 = OrganAMNIST(split='train', download=True, transform=transform_128)
    test_dataset_128 = OrganAMNIST(split='test', download=True, transform=transform_128)
    train_loader_128 = DataLoader(dataset=train_dataset_128, batch_size=64, shuffle=True)
    test_loader_128 = DataLoader(dataset=test_dataset_128, batch_size=64, shuffle=False)

    num_classes = len(np.unique(train_dataset_128.labels))
    cnn_model_128 = CNN128(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn_model_128.parameters(), lr=0.001)

    print("Training Task 7 CNN Model (128x128 Images)")
    start_time = time.time()
    metrics = train_cnn(cnn_model_128, train_loader_128, criterion, optimizer, epochs=10)
    end_time = time.time()

    training_time = end_time - start_time
    print(f"Task 7: Model training completed in {training_time:.2f} seconds.")

    save_metrics_plot(metrics, "Task 7: CNN Training Metrics (128x128)", "Value", "plots/task7_metrics.png")

    accuracy = evaluate_cnn(cnn_model_128, test_loader_128)
    return {"Accuracy": accuracy, "Training Time (seconds)": training_time}


def task8():
    transform_augmented = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    train_dataset_aug = OrganAMNIST(split='train', download=True, transform=transform_augmented)
    test_dataset_aug = OrganAMNIST(split='test', download=True, transform=transform_augmented)
    train_loader_aug = DataLoader(dataset=train_dataset_aug, batch_size=32, shuffle=True)
    test_loader_aug = DataLoader(dataset=test_dataset_aug, batch_size=32, shuffle=False)

    num_classes = len(np.unique(train_dataset_aug.labels))
    transfer_model = TransferLearningModel(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(transfer_model.parameters(), lr=0.001)

    print("Training Task 8 Transfer Learning Model with Augmentation")
    start_time = time.time()
    metrics = train_cnn(transfer_model, train_loader_aug, criterion, optimizer, epochs=10)
    end_time = time.time()

    training_time = end_time - start_time
    print(f"Task 8: Model training completed in {training_time:.2f} seconds.")

    save_metrics_plot(metrics, "Task 8: Transfer Learning Training Metrics", "Value", "plots/task8_metrics.png")

    accuracy = evaluate_cnn(transfer_model, test_loader_aug)
    return {"Accuracy": accuracy, "Training Time (seconds)": training_time}





