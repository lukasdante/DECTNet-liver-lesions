import os
from PIL import Image

def convert_jpg_to_png(root_dir, output_dir):
    """
    Convert all JPG images in a directory to PNG format.
    
    Parameters:
        root_dir (str): The root directory containing JPG images.
        output_dir (str): The directory to save PNG images.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate over all files in the directory
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".jpg"):
                # Build full input file path
                jpg_path = os.path.join(subdir, file)
                
                # Build output PNG path
                png_filename = os.path.splitext(file)[0] + ".png"
                png_path = os.path.join(output_dir, png_filename)
                
                # Open the JPG file and save it as PNG
                try:
                    with Image.open(jpg_path) as img:
                        img.save(png_path, "PNG")
                        print(f"Converted: {jpg_path} -> {png_path}")
                except Exception as e:
                    print(f"Failed to convert {jpg_path}: {e}")

if __name__ == "__main__":
    # Define the input and output directories
    root_directory = "/home/louis/Desktop/dectnet/DECTNet-liver-lesions/Codes/FLLs/dataset/original/jpegs"  # Replace with your directory containing JPG files
    output_directory = "/home/louis/Desktop/dectnet/DECTNet-liver-lesions/Codes/FLLs/dataset/original/pngs"  # Replace with your desired output directory
    
    # Perform the conversion
    convert_jpg_to_png(root_directory, output_directory)
