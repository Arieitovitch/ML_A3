import numpy as np
import matplotlib.pyplot as plt
import os
from medmnist import OrganAMNIST

# Load preprocessed data
data_path = os.path.join("code/processed_data", "mlp_data.npz")

def load_data(file_path):
    """
    Load the preprocessed data from the given file path.
    """
    data = np.load(file_path)
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    return X_train, y_train, X_val, y_val, X_test, y_test

# Load original OrganAMNIST dataset
train_dataset = OrganAMNIST(split="train", download=True)
val_dataset = OrganAMNIST(split="val", download=True)
test_dataset = OrganAMNIST(split="test", download=True)

# Access images and labels
train_images, train_labels = train_dataset.imgs, train_dataset.labels
num_classes_original = train_dataset.info['n_channels']
y_train_original = train_dataset.labels.flatten()
y_val_original = val_dataset.labels.flatten()
y_test_original = test_dataset.labels.flatten()

# Calculate class distributions for the original dataset
train_counts_original = np.bincount(y_train_original, minlength=num_classes_original)
val_counts_original = np.bincount(y_val_original, minlength=num_classes_original)
test_counts_original = np.bincount(y_test_original, minlength=num_classes_original)

# Load data-augmented dataset
X_train_augmented, y_train_augmented, X_val_augmented, y_val_augmented, X_test_augmented, y_test_augmented = load_data(data_path)
num_classes_augmented = len(np.unique(np.concatenate([y_train_augmented, y_val_augmented, y_test_augmented])))

# Calculate class distributions for the data-augmented dataset
train_counts_augmented = np.bincount(y_train_augmented.astype(int), minlength=num_classes_augmented)
val_counts_augmented = np.bincount(y_val_augmented.astype(int), minlength=num_classes_augmented)
test_counts_augmented = np.bincount(y_test_augmented.astype(int), minlength=num_classes_augmented)

# Plot the histograms
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

def add_labels(ax, counts):
    """
    Add data labels to each bar in the bar chart.
    """
    for i, count in enumerate(counts):
        ax.text(i, count + 0.02 * max(counts), str(count), ha='center', fontsize=10, color='black')

# Original training data histogram
axes[0].bar(range(len(train_counts_original)), train_counts_original, color=plt.cm.tab10.colors, align='center')
axes[0].set_title("Original Training Data Distribution")
axes[0].set_xlabel("Classes")
axes[0].set_ylabel("Count")
axes[0].set_xticks(range(len(train_counts_original)))
axes[0].set_ylim(0, max(train_counts_original.max(), train_counts_augmented.max(), test_counts_original.max()) * 1.1)
add_labels(axes[0], train_counts_original)

# Data-augmented training data histogram
axes[1].bar(range(len(train_counts_augmented)), train_counts_augmented, color=plt.cm.tab10.colors, align='center')
axes[1].set_title("Data-Augmented Training Data Distribution")
axes[1].set_xlabel("Classes")
axes[1].set_xticks(range(len(train_counts_augmented)))
add_labels(axes[1], train_counts_augmented)

# Testing data histogram
axes[2].bar(range(len(test_counts_original)), test_counts_original, color=plt.cm.tab10.colors, align='center')
axes[2].set_title("Testing Data Distribution")
axes[2].set_xlabel("Classes")
axes[2].set_xticks(range(len(test_counts_original)))
add_labels(axes[2], test_counts_original)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig("organamnist_combined_distributions.png")
plt.show()
