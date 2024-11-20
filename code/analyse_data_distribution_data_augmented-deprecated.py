import numpy as np
import matplotlib.pyplot as plt
import os

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

# Load the data
X_train, y_train, X_val, y_val, X_test, y_test = load_data(data_path)

# Calculate the number of classes
num_classes = len(np.unique(np.concatenate([y_train, y_val, y_test])))

# Calculate class distributions
train_counts = np.bincount(y_train.astype(int), minlength=num_classes)
val_counts = np.bincount(y_val.astype(int), minlength=num_classes)
test_counts = np.bincount(y_test.astype(int), minlength=num_classes)

# Plot the histograms
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

# Add function to annotate bars
def add_labels(ax, counts):
    """
    Add data labels to each bar in the bar chart.
    """
    for i, count in enumerate(counts):
        ax.text(i, count + 0.02 * max(counts), str(count), ha='center', fontsize=10, color='black')

# Training data histogram
axes[0].bar(range(len(train_counts)), train_counts, color=plt.cm.tab10.colors, align='center')
axes[0].set_title("Distribution of Training Data")
axes[0].set_xlabel("Classes")
axes[0].set_ylabel("Count")
axes[0].set_xticks(range(len(train_counts)))
axes[0].set_ylim(0, max(train_counts.max(), test_counts.max()) * 1.1)
add_labels(axes[0], train_counts)  # Add data labels to the bars

# Testing data histogram
axes[1].bar(range(len(test_counts)), test_counts, color=plt.cm.tab10.colors, align='center')
axes[1].set_title("Distribution of Testing Data")
axes[1].set_xlabel("Classes")
axes[1].set_xticks(range(len(test_counts)))
add_labels(axes[1], test_counts)  # Add data labels to the bars

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig("organamnist_class_distributions_with_labels_data_augmented.png")
plt.show()
