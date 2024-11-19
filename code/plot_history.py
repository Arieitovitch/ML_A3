import pickle
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

def plot_history(training_history, validation_history=None, title="Loss history", save_path=None, ax=None):
    """
    Plot the history of loss during training and validation.
    Annotate the optimal epoch based on validation loss.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot training and validation losses
    ax.plot(list(training_history.keys()), list(training_history.values()), label='Training Loss')
    if validation_history is not None:
        ax.plot(list(validation_history.keys()), list(validation_history.values()), label='Validation Loss')

        # Find the optimal epoch and validation loss
        optimal_epoch = min(validation_history, key=validation_history.get)
        optimal_loss = validation_history[optimal_epoch]
        
        # Annotate the optimal epoch
        ax.annotate(
            f"Optimal Epoch: {optimal_epoch}\nLoss: {optimal_loss:.4f}",
            xy=(optimal_epoch, optimal_loss),
            xytext=(optimal_epoch + 2, optimal_loss + 0.1),  # Adjust text position
            arrowprops=dict(facecolor='black', arrowstyle="->"),
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white")
        )

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    if save_path is not None:
        plt.savefig(save_path)

def plot_name(model_name, prefix = "Loss of Model - "):
    mapping = {
        "Model 1 (No Hidden Layers)": "No Hidden Layer",
        "Model 2 (One Hidden Layer)": "One Hidden Layer",
        "Model 3 (Two Hidden Layers)": "Two Hidden Layers",
        "Model with Tanh Activations" : "TanH Activation",
        "Model with Leaky ReLU Activations" : "Leaky ReLU Activation",
        "Model with L1 Regularization": "L1 Regularization",
        "Model with L2 Regularization": "L2 Regularization",
        "Model on Unnormalized Data" : "Unnormalized Data",
    }
    return prefix + mapping.get(model_name, model_name)

def load_history(path):
    """
    Load the history of loss from a pickle file.
    """
    with open(path, "rb") as f:
        history = pickle.load(f)
    return history


# Plotting Loss Histories with Subplots and Overlaid Validation Losses
tasks = ['task1', 'task2', 'task3', 'task4']

for task in tasks:
    training_histories = load_history(f"histories/loss_histories_{task}.pkl")
    validation_histories = load_history(f"histories/val_loss_histories_{task}.pkl")
    num_models = len(training_histories)
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5), squeeze=False)
    axes = axes.flatten()
    # print(type(training_histories))
    # print(training_histories)
    for i, (model_name, history) in enumerate(training_histories.items()):
        training_history = history
        validation_history = validation_histories[model_name]
        title = plot_name(model_name)#f"{task.capitalize()} - Model {i+1}"
        plot_history(
            training_history, validation_history, title=title, save_path=None, ax=axes[i]
        )
    plt.tight_layout()
    plt.savefig(f"figures/{task}_losses.png")
    # plt.show()

def load_accuracies(path):
    """
    Load accuracies from a pickle file.
    """
    with open(path, "rb") as f:
        accuracies = pickle.load(f)
    return accuracies


def build_table1_4(accuracies):
    columns = []
    row_values = []
    for task in accuracies:
        columns.extend(accuracies[task].keys())
        row_values.extend(accuracies[task].values())
    
    columns = list(map(lambda x : plot_name(x, ""), columns))
    row_values = list(map(lambda x : f"{round(x * 100, 2) }%", row_values))
    df = pd.DataFrame([row_values], columns=columns, index=['Accuracy'])
    
    return df.transpose()

accuracies = load_accuracies("all_task_accuracies.pkl")
df = build_table1_4(accuracies)
# Create a Matplotlib figure
fig, ax = plt.subplots(figsize=(10, 2))  # Adjust figsize as needed

# Hide the axes
ax.axis('tight')
ax.axis('off')

# Create the table
table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc='center', cellLoc='center')

# Adjust the table properties
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(df.columns))))

# Show the plot
plt.savefig("figures/accuracies_1_4.png")