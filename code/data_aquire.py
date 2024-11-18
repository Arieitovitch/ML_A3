import numpy as np
import os
from medmnist import OrganAMNIST# type: ignore
from medmnist import INFO# type: ignore

def normalize_data(data):
    """
    Normalize the pixel values of the dataset to be in the range [0, 1].
    """
    return data / 255.0

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
    """
    Main function to acquire and preprocess the OrganAMNIST dataset.
    Creates two formats:
    - MLP format: Flattened and normalized.
    - CNN format: 2D normalized.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load the dataset
    train_dataset = OrganAMNIST(split="train", download=True)
    val_dataset = OrganAMNIST(split="val", download=True)
    test_dataset = OrganAMNIST(split="test", download=True)

    # Prepare MLP format (flattened)
    X_train_mlp = flatten_data(normalize_data(train_dataset.imgs))
    y_train = train_dataset.labels.flatten()
    X_val_mlp = flatten_data(normalize_data(val_dataset.imgs))
    y_val = val_dataset.labels.flatten()
    X_test_mlp = flatten_data(normalize_data(test_dataset.imgs))
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

    # Prepare CNN format (normalized 2D images)
    X_train_cnn = normalize_data(train_dataset.imgs)
    X_val_cnn = normalize_data(val_dataset.imgs)
    X_test_cnn = normalize_data(test_dataset.imgs)

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
