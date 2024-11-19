import pickle
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_history(history, title="Loss history", save_path=None):
    """
    Plot the history of loss during training.
    :param history: Dictionary of epoch: loss
    :param title: Title of the plot
    :param save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    plt.plot(list(history.keys()), list(history.values()))
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    if save_path is not None:
        plt.savefig(save_path)
    # plt.show()

def load_history(path):
    """
    Load the history of loss from a pickle file.
    :param path: Path to the pickle file
    :return: Dictionary of epoch: loss
    """
    with open(path, "rb") as f:
        history = pickle.load(f)
    return history

# Load the history of loss for Task 1

histories = list(load_history("histories/loss_histories_task1.pkl").values())
for i, history in enumerate(histories):
    print(history)
    print(type(history))
    plot_history(history, title="Loss history for Task 1", save_path=f"loss_task1_{1+i}.png")

# Load the history of loss for Task 2
histories = list(load_history("histories/loss_histories_task2.pkl").values())
for i, history in enumerate(histories):
    plot_history(history, title="Loss history for Task 2", save_path=f"loss_task2_{1+i}.png")

# Load the history of loss for Task 3
histories = list(load_history("histories/loss_histories_task3.pkl").values())
for i, history in enumerate(histories):
    plot_history(history, title="Loss history for Task 3", save_path=f"loss_task3_{1+i}.png")

# Load the history of loss for Task 4
histories = list(load_history("histories/loss_histories_task4.pkl").values())
for i, history in enumerate(histories):
    plot_history(history, title="Loss history for Task 4", save_path=f"loss_task4_{1+i}.png")