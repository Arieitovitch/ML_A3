import torch
import os
os.environ["MKL_NUM_THREADS"] = "1"
print(torch.cuda.is_available())  # Should print True
print(torch.cuda.device_count())  # Number of GPUs available
print(torch.cuda.get_device_name(0))  # Name of the first GPU

# Get the current directory
current_directory = os.getcwd()

# Walk through the directory tree
for dirpath, dirnames, filenames in os.walk(current_directory):
    print(f"Directory: {dirpath}")
    for dirname in dirnames:
        if not dirname.startswith("."):
            print(f"  Folder: {dirname}")
    for filename in filenames:
        if not filename.startswith("."):
            print(f"  File: {filename}")