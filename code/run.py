# this file is going to be run by slurm
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

def adjust_labels(labels):
    labels = labels.squeeze()
    if labels.dim() == 0:
        labels = labels.unsqueeze(0)
    return labels.long()

class TransferLearningModel(nn.Module):
    def __init__(self, num_classes):
        super(TransferLearningModel, self).__init__()
        # Load a pre-trained ResNet model
        self.model = models.resnet18(pretrained=True)
        # Modify the first convolutional layer to accept 1-channel images
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Replace the final fully connected layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

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
    data = np.load(os.path.join("processed_data", "mlp_data.npz"))
    X_train = data['X_train'] * 255  # Reverting normalization
    y_train = data['y_train']
    X_test = data['X_test'] * 255
    y_test = data['y_test']
    return X_train, y_train, X_test, y_test

data_path = os.path.join("processed_data", "mlp_data.npz")
X_train, y_train, X_val, y_val, X_test, y_test = load_data(data_path)
input_size = X_train.shape[1]
hidden_layers = []
# Determine the number of classes
num_classes = len(np.unique(y_train))

# One-hot encode the labels
y_train_one_hot = one_hot_encode(y_train, num_classes)
y_test_one_hot = one_hot_encode(y_test, num_classes)

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

def train_cnn(model, train_loader, criterion, optimizer, epochs=10, save_weights=True, path_prefix=""):
    history = {}
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            # Zero gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(images)
            labels = labels.view(-1).long()
            loss = criterion(outputs, labels)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        loss = float(f"{total_loss/len(train_loader):.4f}")
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss}")
        history[epoch] = loss
    if save_weights:
        torch.save(model.state_dict(), f"weights/{path_prefix + '/' if path_prefix else ''}weights.pth")
    return history
    
        
# Evaluate the model
def evaluate_cnn(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.squeeze().long()).sum().item()
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return round(accuracy,4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        torch.save(model.state_dict(), f"weights/{path_prefix + '/' if path_prefix else ''}weights.pth")
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
def train_and_evaluate(model, model_name):
    print(f"Training {model_name}")
    model.fit(X_train, y_train_one_hot, epochs=50, lr=0.01, batch_size=64, save_weights=True, path_prefix=model_name)
    y_pred = model.predict(X_test)
    test_accuracy = np.mean(y_pred == y_test)
    print(f"{model_name} Test Accuracy: {test_accuracy * 100:.2f}%\n")
    return test_accuracy


def task1():
    # Model 1: No hidden layers
    activations = ['softmax']  # Only output activation
    model1 = MLP(input_size, [], num_classes, activations)
    acc1 = train_and_evaluate(model1, "Model 1 (No Hidden Layers)")

    # Model 2: One hidden layer
    hidden_layers = [256]
    activations = ['relu', 'softmax']
    model2 = MLP(input_size, hidden_layers, num_classes, activations)
    acc2 = train_and_evaluate(model2, "Model 2 (One Hidden Layer)")

    # Model 3: Two hidden layers
    hidden_layers = [256, 256]
    activations = ['relu', 'relu', 'softmax']
    model3 = MLP(input_size, hidden_layers, num_classes, activations)
    acc3 = train_and_evaluate(model3, "Model 3 (Two Hidden Layers)")
    
    # Write histories to memory
    histories = {
        "Model 1 (No Hidden Layers)": model1.history,
        "Model 2 (One Hidden Layer)": model2.history,
        "Model 3 (Two Hidden Layers)": model3.history,
    }
    # Save to a pickle file
    with open("histories/loss_histories_task1.pkl", "wb") as pkl_file:
        pickle.dump(histories, pkl_file)
    
    return {
        "Model 1 (No Hidden Layers)": acc1, 
        "Model 2 (One Hidden Layer)": acc2, 
        "Model 3 (Two Hidden Layers)": acc3,
    }
    
    
def task2():
    # Model with tanh activations
    activations_tanh = ['tanh', 'tanh', 'softmax']
    model_tanh = MLP(input_size, hidden_layers, num_classes, activations_tanh)
    acc_tanh = train_and_evaluate(model_tanh, "Model with Tanh Activations")

    # Model with Leaky ReLU activations
    activations_leaky_relu = ['leaky_relu', 'leaky_relu', 'softmax']
    model_leaky_relu = MLP(input_size, hidden_layers, num_classes, activations_leaky_relu)
    acc_leaky_relu = train_and_evaluate(model_leaky_relu, "Model with Leaky ReLU Activations")
    
    # Write histories to memory
    histories = {
        "Model with Tanh Activations": model_tanh.history,
        "Model with Leaky ReLU Activations": model_leaky_relu.history,
    }
    # Save to a pickle file
    with open("histories/loss_histories_task2.pkl", "wb") as pkl_file:
        pickle.dump(histories, pkl_file)
    
    return {
        "Model with Tanh Activations": acc_tanh,
        "Model with Leaky ReLU Activations": acc_leaky_relu,
    }
    
def task3():
    activations = ['leaky_relu', 'leaky_relu', 'softmax']
    model_l1 = MLPREG(input_size, hidden_layers, num_classes, activations)
    print("Training Model with L1 Regularization")
    model_l1.fit(X_train, y_train_one_hot, epochs=50, lr=0.01, batch_size=64, l1_lambda=0.001)
    y_pred_l1 = model_l1.predict(X_test)
    acc_l1 = np.mean(y_pred_l1 == y_test)
    print(f"Model with L1 Regularization Test Accuracy: {acc_l1 * 100:.2f}%\n")
    
    model_l2 = MLPREG(input_size, hidden_layers, num_classes, activations)
    print("Training Model with L2 Regularization")
    
    model_l2.fit(X_train, y_train_one_hot, epochs=50, lr=0.01, batch_size=64, l2_lambda=0.001)
    y_pred_l2 = model_l2.predict(X_test)
    acc_l2 = np.mean(y_pred_l2 == y_test)
    print(f"Model with L2 Regularization Test Accuracy: {acc_l2 * 100:.2f}%\n")
    
    # Write histories to memory
    histories = {
        "Model with L1 Regularization": model_l1.history,
        "Model with L2 Regularization": model_l2.history,
    }
    # Save to a pickle file
    with open("histories/loss_histories_task3.pkl", "wb") as pkl_file:
        pickle.dump(histories, pkl_file)
    
    return {
        "Model with L1 Regularization": acc_l1,
        "Model with L2 Regularization": acc_l2,
    }
    
def task4():
    activations = ['leaky_relu', 'leaky_relu', 'softmax']
    model_un = MLP(input_size, hidden_layers, num_classes, activations)

    # Train the model on unnormalized data
    print("Training Model on Unnormalized Data")
    model_un.fit(X_train_un, y_train_un_one_hot, epochs=50, lr=0.01, batch_size=64)
    y_pred_un = model_un.predict(X_test_un)
    acc_un = np.mean(y_pred_un == y_test_un)
    print(f"Model on Unnormalized Data Test Accuracy: {acc_un * 100:.2f}%\n")
    
    # Write histories to memory
    histories = {
        "Model on Unnormalized Data": model_un.history,
    }
    # Save to a pickle file
    with open("histories/loss_histories_task4.pkl", "wb") as pkl_file:
        pickle.dump(histories, pkl_file)

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
    model_128.fit(X_train_128, y_train_128_one_hot, epochs=50, lr=0.01, batch_size=64)
    
    # Evaluate the model
    y_pred_128 = model_128.predict(X_test_128)
    acc_128 = np.mean(y_pred_128 == y_test_128)
    print(f"MLP on 128-Pixel Images Test Accuracy: {acc_128 * 100:.2f}%\n")
    
    # Write histories to memory
    histories = {
        "MLP on 128-Pixel Images": model_128.history,
    }
    # Save to a pickle file
    with open("histories/loss_histories_task5.pkl", "wb") as pkl_file:
        pickle.dump(histories, pkl_file)
    
    return {
        "MLP on 128-Pixel Images": acc_128,
    }

def task6():
    num_classes = len(np.unique(train_dataset.labels))

    # Initialize the model
    cnn_model = CNN(num_classes)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
    train_cnn(cnn_model, train_loader, criterion, optimizer, epochs=10, save_weights=True, path_prefix="task6")
    acc = evaluate_cnn(cnn_model, test_loader) 
    
    return {
        "CNN": acc,
    }
    
def task7():
    transform_128 = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    # Load the datasets with transformations
    train_dataset_128 = OrganAMNIST(split='train', download=True, transform=transform_128)
    test_dataset_128 = OrganAMNIST(split='test', download=True, transform=transform_128)
    
    # Create DataLoaders
    batch_size = 64
    train_loader_128 = DataLoader(dataset=train_dataset_128, batch_size=batch_size, shuffle=True)
    test_loader_128 = DataLoader(dataset=test_dataset_128, batch_size=batch_size, shuffle=False)
    
    # Determine the number of classes
    num_classes = len(np.unique(train_dataset_128.labels))
    
    # Initialize the CNN model for 128x128 images
    cnn_model_128 = CNN128(num_classes)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn_model_128.parameters(), lr=0.001)
    
    # Train the model
    print("Training CNN on 128-Pixel Images")
    train_cnn(cnn_model_128, train_loader_128, criterion, optimizer, epochs=10, save_weights=True, path_prefix="task7")
    
    # Evaluate the model
    print("Evaluating CNN on 128-Pixel Images")
    acc = evaluate_cnn(cnn_model_128, test_loader_128)
    
    return {
        "CNN on 128-Pixel Images": acc,
    }
    
def task8():
    transform_augmented = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize to [-1, 1]
    ])
    train_dataset_aug = OrganAMNIST(split='train', download=True, transform=transform_augmented)
    test_dataset_aug = OrganAMNIST(split='test', download=True, transform=transform_augmented)
    batch_size = 32  
    train_loader_aug = DataLoader(dataset=train_dataset_aug, batch_size=batch_size, shuffle=True)
    test_loader_aug = DataLoader(dataset=test_dataset_aug, batch_size=batch_size, shuffle=False)
    num_classes = len(np.unique(train_dataset_aug.labels))
    transfer_model = TransferLearningModel(num_classes)
    transfer_model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(transfer_model.parameters(), lr=0.001)
    print("Training Transfer Learning Model with Data Augmentation")
    train_transfer_model(transfer_model, train_loader_aug, criterion, optimizer, epochs=10, save_weights=True, path_prefix="task8")
    print("Evaluating Transfer Learning Model")
    acc = evaluate_transfer_model(transfer_model, test_loader_aug)
    return {
        "Transfer Learning Model with Data Augmentation": acc,
    }
    

results = {
    "Task 1": task1(),
    "Task 2": task2(),
    "Task 3": task3(),
    "Task 4": task4(),
    "Task 5": task5(),
    "Task 6": task6(),
    "Task 7": task7(),
    "Task 8": task8(),
}

# Save results to a pickle file
with open("all_task_accuracies.pkl", "wb") as pkl_file:
    pickle.dump(results, pkl_file)