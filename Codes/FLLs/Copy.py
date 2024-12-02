import os
import shutil
from glob import glob

def get_file_matches(directory_a, directory_b):
    """
    Collects files from directory_a (starting with gt_) and matches them to directory_b files.
    """
    # Get all files from directory_a and directory_b
    files_a = glob(os.path.join(directory_a, "gt_*"))
    files_b = glob(os.path.join(directory_b, "*"))
    
    # Create a dictionary of filenames (without gt_) from directory_a for matching
    match_dict = {}
    for file_a in files_a:
        # Remove 'gt_' from the beginning of the file name to match with files in directory_b
        base_name_a = os.path.basename(file_a)[3:]  # Remove 'gt_' from the filename
        match_dict[base_name_a] = file_a
    
    # Now match files from directory_b with those in directory_a (after removing 'gt_')
    matched_files = []
    for file_b in files_b:
        base_name_b = os.path.basename(file_b)
        if base_name_b in match_dict:
            matched_files.append((match_dict[base_name_b], file_b))  # Add the matching pair (from directory_a, from directory_b)
    
    return matched_files

def copy_files(matched_files, destination_directory):
    """
    Copies matched files from directory_b to destination_directory.
    """
    os.makedirs(destination_directory, exist_ok=True)
    
    for file_a, file_b in matched_files:
        # Copy the file from directory_b to the destination directory
        dest_path = os.path.join(destination_directory, os.path.basename(file_b))
        shutil.copy(file_b, dest_path)
        print(f"Copied: {file_b} -> {dest_path}")

if __name__ == "__main__":
    source_directory = "/home/louis/Desktop/dectnet/Codes/FLLs/dataset/original/pngs"
    directory_test_gt = "/home/louis/Desktop/dectnet/Codes/FLLs/dataset/root/TestDataset/gt"
    directory_train_gt = "/home/louis/Desktop/dectnet/Codes/FLLs/dataset/root/TrainDataset/gt"
    destination_test_directory = "/home/louis/Desktop/dectnet/Codes/FLLs/dataset/root/TestDataset/images"  
    destination_train_directory = "/home/louis/Desktop/dectnet/Codes/FLLs/dataset/root/TrainDataset/images" 

    # Get matched files from both directories
    train_matched_files = get_file_matches(directory_train_gt, source_directory)
    test_matched_files = get_file_matches(directory_test_gt, source_directory)
    
    print(f"Found {len(train_matched_files)} train matching files.")
    print(f"Found {len(test_matched_files)} test matching files.")
    
    # Copy the matched files from directory_b to the destination directory
    copy_files(train_matched_files, destination_train_directory)
    copy_files(test_matched_files, destination_test_directory)