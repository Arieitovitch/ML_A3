import numpy as np
import matplotlib.pyplot as plt
from medmnist import OrganAMNIST

# Load OrganAMNIST dataset
train_dataset = OrganAMNIST(split="train", download=True)
val_dataset = OrganAMNIST(split="val", download=True)
test_dataset = OrganAMNIST(split="test", download=True)

# Extract the number of classes
num_classes = train_dataset.info['n_channels']

# Extract labels
y_train = train_dataset.labels.flatten()
y_val = val_dataset.labels.flatten()
y_test = test_dataset.labels.flatten()

# Combine train and validation sets if needed
y_train_val = np.concatenate([y_train, y_val])

# Calculate class distributions
train_counts = np.bincount(y_train, minlength=num_classes)
val_counts = np.bincount(y_val, minlength=num_classes)
test_counts = np.bincount(y_test, minlength=num_classes)

# Plot the histograms
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

# Training data histogram
axes[0].bar(range(len(train_counts)), train_counts, color=plt.cm.tab10.colors, align='center')
axes[0].set_title("Distribution of Training Data")
axes[0].set_xlabel("Classes")
axes[0].set_ylabel("Count")
axes[0].set_xticks(range(len(train_counts)))
axes[0].set_ylim(0, max(train_counts.max(), test_counts.max()) * 1.1)

# Testing data histogram
axes[1].bar(range(len(test_counts)), test_counts, color=plt.cm.tab10.colors, align='center')
axes[1].set_title("Distribution of Testing Data")
axes[1].set_xlabel("Classes")
axes[1].set_xticks(range(len(test_counts)))

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig("organamnist_class_distributions.png")
plt.show()
