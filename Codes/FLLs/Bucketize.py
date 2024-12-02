import os
import random
import re
import shutil
from collections import defaultdict
from glob import glob

def collect_files(root_dir):
    """
    Collect all ground truth files from the root directory matching the naming pattern.
    """
    # Updated regex pattern to handle dataset numbers of any length
    pattern = re.compile(r"^gt_DataSet\d+_(Normal|Abscess|Cyst|Tumor)_jpg\.rf\.[a-zA-Z0-9]+\.jpg\.png$")
    
    # Use glob to recursively find all files
    all_files = glob(os.path.join(root_dir, "**", "*.png"), recursive=True)
    
    # Filter files that match the regex pattern
    matched_files = [f for f in all_files if pattern.search(os.path.basename(f))]
    return matched_files

def bucketize_labels(file_paths, train_ratio=0.8):
    """
    Bucketize the ground truth labels into training and testing sets with a given ratio.
    """
    type_buckets = defaultdict(list)
    
    # Categorize files by Type
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        parts = file_name.split("_")
        file_type = parts[2]  # Extract the `Type`
        type_buckets[file_type].append(file_path)
    
    # Split into training and testing sets
    train_set = []
    test_set = []
    
    for file_type, files in type_buckets.items():
        random.shuffle(files)  # Shuffle files for randomness
        split_index = int(len(files) * train_ratio)  # Calculate split index
        
        train_set.extend(files[:split_index])
        test_set.extend(files[split_index:])
    
    return train_set, test_set

def copy_and_rename(files, destination_dir):
    """
    Copy and rename files to the destination directory.
    """
    os.makedirs(destination_dir, exist_ok=True)
    
    for file_path in files:
        file_name = os.path.basename(file_path)
        
        # Rename file: Remove the `.jpg` before `.png`
        new_name = re.sub(r'\.jpg\.png$', '.png', file_name)
        
        # Destination path
        dest_path = os.path.join(destination_dir, new_name)
        
        # Copy file
        shutil.copy(file_path, dest_path)
        print(f"Copied and renamed: {file_path} -> {dest_path}")

if __name__ == "__main__":
    # Root directory where all images are stored
    root_directory = "/home/louis/Desktop/dectnet/Codes/FLLs/dataset/original/source/muticlass_oneshot_masks_fixed"
    
    # Output directories for training and testing sets
    train_directory = "/home/louis/Desktop/dectnet/Codes/FLLs/dataset/root/TrainDataset/gt"
    test_directory = "/home/louis/Desktop/dectnet/Codes/FLLs/dataset/root/TestDataset/gt"
    
    # Collect all matching files
    files = collect_files(root_directory)
    print(f"Total files collected: {len(files)}")
    
    # Bucketize into training and testing sets
    train, test = bucketize_labels(files)
    
    # Print dataset statistics
    print(f"Training Set: {len(train)}")
    print(f"Testing Set: {len(test)}")
    
    # Copy and rename training and testing sets
    print("\nProcessing Training Set...")
    copy_and_rename(train, train_directory)
    
    print("\nProcessing Testing Set...")
    copy_and_rename(test, test_directory)
