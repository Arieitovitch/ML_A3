import numpy as np
import os
from medmnist import OrganAMNIST# type: ignore
from medmnist import INFO# type: ignore
import random
from scipy.ndimage import rotate

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

def analyze_data_distribution(data, num_classes):
    """
    Calculate the class distribution of the dataset.
    """
    return np.bincount(data, minlength=num_classes)

def data_augmentation(dataset, class_counts, min_samples=6164):
    """
    Apply data augmentation to balance the dataset by adding synthetic samples
    through random rotations and flips for underrepresented classes.

    Args:
        dataset: The original dataset to augment (e.g., OrganAMNIST train dataset).
        class_counts: A list of the counts of each class in the dataset.
        min_samples: The minimum number of samples required for each class.

    Returns:
        Augmented dataset with additional synthetic samples for underrepresented classes.
    """
    augmented_imgs = []
    augmented_labels = []
    
    # Get existing images and labels
    imgs = dataset.imgs
    labels = dataset.labels.flatten()
    
    for class_id in range(len(class_counts)):
        current_count = class_counts[class_id]
        
        # If the class already has sufficient samples, skip it
        if current_count >= min_samples:
            continue
        
        # Get all images for the current class
        class_imgs = imgs[labels == class_id]
        num_to_add = min_samples - current_count
        
        for _ in range(num_to_add):
            # Randomly select an image from the current class
            img = random.choice(class_imgs)
            
            # Apply random rotation
            angle = random.choice([0, 90, 180, 270])
            rotated_img = rotate(img, angle, reshape=False, mode='nearest')
            
            # Apply random flip
            if random.choice([True, False]):
                flipped_img = np.flip(rotated_img, axis=1)  # Horizontal flip
            else:
                flipped_img = np.flip(rotated_img, axis=0)  # Vertical flip
            
            # Add the augmented image and label to the lists
            augmented_imgs.append(flipped_img)
            augmented_labels.append(class_id)
    
    # Combine original data with augmented data
    augmented_imgs = np.array(augmented_imgs)
    augmented_labels = np.array(augmented_labels)
    
    combined_imgs = np.concatenate((imgs, augmented_imgs), axis=0)
    combined_labels = np.concatenate((labels, augmented_labels), axis=0)
    
    # Return the new dataset structure
    dataset.imgs = combined_imgs
    dataset.labels = combined_labels.reshape(-1, 1)
    return dataset

def main(output_dir="code/processed_data"):
    os.makedirs(output_dir, exist_ok=True)

    # Load the dataset
    train_dataset = OrganAMNIST(split="train", download=True)
    val_dataset = OrganAMNIST(split="val", download=True)
    test_dataset = OrganAMNIST(split="test", download=True)

    # Extract the number of classes
    num_classes = train_dataset.info['n_channels']

    # Analyze the class distribution of the dataset
    train_counts = analyze_data_distribution(train_dataset.labels.flatten(), num_classes)
    print(train_counts)

    # Data Augmentation of categories with less than specified number of samples
    new_train_dataset = data_augmentation(train_dataset, train_counts)

    train_counts = analyze_data_distribution(new_train_dataset.labels.flatten(), num_classes)
    print(train_counts)
    train_dataset = new_train_dataset

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
        os.path.join(output_dir, "mlp_data_unnormalized.npz"),
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
