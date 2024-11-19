import numpy as np
import os
from medmnist import OrganAMNIST# type: ignore
from medmnist import INFO# type: ignore

def normalize_data_to_01(data):
    """
    Normalize the pixel values of the dataset to be in the range [0, 1].
    """
    return data / 255.0

def normalize_data(data, mean=None, std=None):
    """
    Standardize the dataset to have zero mean and unit variance.
    """
    if mean is None or std is None:
        mean = np.mean(data)
        std = np.std(data)
    if std == 0:
        std = 1
    data_normalized = (data - mean) / std
    return data_normalized, mean, std

def flatten_data(data):
    """
    Flatten the images for MLP input.
    """
    return data.reshape(data.shape[0], -1)

def save_data(file_name, data):
    """
    Save the processed data to a file for reuse.
    """
    np.savez_compressed(file_name, **data)

def main(output_dir="processed_data"):
    os.makedirs(output_dir, exist_ok=True)

    # Load the dataset
    train_dataset = OrganAMNIST(split="train", download=True)
    val_dataset = OrganAMNIST(split="val", download=True)
    test_dataset = OrganAMNIST(split="test", download=True)

    # Compute mean and std from training data
    X_train_normalized, mean, std = normalize_data(train_dataset.imgs)
    X_val_normalized, _, _ = normalize_data(val_dataset.imgs, mean, std)
    X_test_normalized, _, _ = normalize_data(test_dataset.imgs, mean, std)

    # Prepare MLP format (standardized and flattened) ----------------------
    X_train_mlp = flatten_data(X_train_normalized)
    y_train = train_dataset.labels.flatten()
    X_val_mlp = flatten_data(X_val_normalized)
    y_val = val_dataset.labels.flatten()
    X_test_mlp = flatten_data(X_test_normalized)
    y_test = test_dataset.labels.flatten()

    # Save MLP data
    save_data(
        os.path.join(output_dir, "mlp_data.npz"),
        {
            "X_train": X_train_mlp,
            "y_train": y_train,
            "X_val": X_val_mlp,
            "y_val": y_val,
            "X_test": X_test_mlp,
            "y_test": y_test,
        }
    )

    # Create un-normalized data for task 4 ----------------------
    X_train_unnormalized = train_dataset.imgs
    X_val_unnormalized = val_dataset.imgs
    X_test_unnormalized = test_dataset.imgs
    
    # Prepare MLP format (flattened)
    X_train_mlp = flatten_data(X_train_unnormalized)
    X_val_mlp = flatten_data(X_val_unnormalized)
    X_test_mlp = flatten_data(X_test_unnormalized)
    y_train = train_dataset.labels.flatten()
    y_val = val_dataset.labels.flatten()
    y_test = test_dataset.labels.flatten()

    # Save MLP data
    save_data(
        os.path.join(output_dir, "mlp_data.npz"),
        {
            "X_train": X_train_mlp,
            "y_train": y_train,
            "X_val": X_val_mlp,
            "y_val": y_val,
            "X_test": X_test_mlp,
            "y_test": y_test,
        }
    )

    # Prepare CNN format (normalized 2D images) ----------------------
    X_train_cnn = normalize_data_to_01(train_dataset.imgs)
    X_val_cnn = normalize_data_to_01(val_dataset.imgs)
    X_test_cnn = normalize_data_to_01(test_dataset.imgs)

    # Save CNN data
    save_data(
        os.path.join(output_dir, "cnn_data.npz"),
        {
            "X_train": X_train_cnn,
            "y_train": y_train,
            "X_val": X_val_cnn,
            "y_val": y_val,
            "X_test": X_test_cnn,
            "y_test": y_test,
        }
    )

    print(f"Data saved to {output_dir}!")

if __name__ == "__main__":
    main()
