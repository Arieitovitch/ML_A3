import pickle
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

def plot_history(training_history, validation_history=None, accuracy_history=None, title="Loss history", save_path=None, ax=None):
    """
    Plot the history of loss during training and validation, and overlay accuracy on a secondary y-axis.
    Annotate the optimal epoch based on validation loss.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Plot training and validation losses on the primary y-axis
    ax.plot(list(training_history.keys()), list(training_history.values()), label='Training Loss', color='blue')
    if validation_history is not None:
        ax.plot(list(validation_history.keys()), list(validation_history.values()), label='Validation Loss', color='orange')

        # Find the optimal epoch and validation loss
        optimal_epoch = min(validation_history, key=validation_history.get)
        optimal_loss = validation_history[optimal_epoch]
        
        # Annotate the optimal epoch
        # ax.annotate(
        #     f"Epoch: {optimal_epoch}\nLoss: {optimal_loss:.4f}",
        #     xy=(optimal_epoch, optimal_loss),
        #     xytext=(optimal_epoch + -1, optimal_loss + 0.1),  # Adjust text position
        #     arrowprops=dict(facecolor='black', arrowstyle="->"),
        #     fontsize=10,
        #     bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white")
        # )

    # Configure primary y-axis
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper left")

    # Create a secondary y-axis for accuracy if provided
    if accuracy_history is not None:
        ax2 = ax.twinx()
        ax2.plot(list(accuracy_history.keys()), list(accuracy_history.values()), label='Accuracy', color='green', linestyle='--')
        ax2.set_ylabel("Accuracy (%)")
        ax2.legend(loc="upper right")
        ax2.grid(False)  # Turn off the grid for the secondary y-axis

    # Save the figure if a path is provided
    if save_path is not None:
        plt.savefig(save_path)
    # plt.show()

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
    accuracy_histories = load_history(f"histories/accuracy_histories_{task}.pkl")
    
    num_models = len(training_histories)
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5), squeeze=False)
    axes = axes.flatten()
    # print(type(training_histories))
    # print(training_histories)
    for i, (model_name, history) in enumerate(training_histories.items()):
        training_history = history
        validation_history = validation_histories[model_name]
        accuracy_history = accuracy_histories[model_name]
        accuracy_history = {k: round(v * 100,2) for k, v in accuracy_history.items()}  # Convert accuracy to percentage
        title = plot_name(model_name)#f"{task.capitalize()} - Model {i+1}"
        plot_history(
            training_history, validation_history, accuracy_history, 
            title=title, save_path=None, ax=axes[i]
        )
    plt.tight_layout()
    plt.savefig(f"figures/{task}_losses.png")
    # plt.show()




# print all accuracies (final values) table
def load_all_accuracies(path):
    """
    Load accuracies from a pickle file.
    """
    with open(path, "rb") as f:
        accuracies = pickle.load(f)
    return accuracies

def build_table1_4(accuracies, validation_histories, accuracy_histories):
    columns = []
    final_accuracy_values = []
    optimal_epoch_accuracies = []
    
    # Iterate over tasks
    for task in accuracies:
        for model_name in accuracies[task].keys():
            # Add model names to columns
            columns.append(plot_name(model_name, ""))
            
            # Get final accuracy
            final_accuracy = accuracies[task][model_name]
            final_accuracy_values.append(f"{round(final_accuracy * 100, 2)}%")
            
            # Get the optimal epoch from validation histories
            task_s = task.replace(" ", "").lower()
            validation_history = validation_histories[task_s][model_name]
            optimal_epoch = min(validation_history, key=validation_history.get)
            
            # Get accuracy at the optimal epoch
            accuracy_history = accuracy_histories[task_s][model_name]
            optimal_accuracy = accuracy_history[optimal_epoch]
            optimal_epoch_accuracies.append(f"{round(optimal_accuracy * 100, 2)}%")
    
    # Create the DataFrame
    df = pd.DataFrame(
        {
            "Final Accuracy": final_accuracy_values,
            "Accuracy at Optimal Epoch": optimal_epoch_accuracies
        },
        index=columns
    )
    
    return df

# Load histories
accuracies = load_all_accuracies("all_task_accuracies.pkl")
validation_histories = {task: load_history(f"histories/val_loss_histories_{task}.pkl") for task in tasks}
accuracy_histories = {task: load_history(f"histories/accuracy_histories_{task}.pkl") for task in tasks}

# Build the table
df = build_table1_4(accuracies, validation_histories, accuracy_histories)

# Create a Matplotlib figure
fig, ax = plt.subplots(figsize=(12, 4))  # Adjust figsize as needed

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
plt.savefig("figures/accuracies_with_optimal_epoch.png", bbox_inches='tight')
