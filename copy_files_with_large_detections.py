import os
import shutil
from PIL import Image


def copy_files_with_small_detections(source_dir, target_dir, pixel_threshold=100):
    # Create target directory if it does not exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Iterate through all files in the source directory
    for filename in os.listdir(source_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(source_dir, filename)
            image_path = os.path.join(source_dir,
                                      filename.replace('.txt', '.png'))  # Change to match your image extension
            cad_path = os.path.join(source_dir,
                                      filename.replace('Image.txt', 'Cad.png'))

            # Check if the corresponding image file exists
            if os.path.exists(image_path):
                # Open image to get its dimensions
                with Image.open(image_path) as img:
                    width, height = img.size

                # Read detections from the txt file
                with open(filepath, 'r') as file:
                    for line in file:
                        _, x_center, y_center, w, h = map(float, line.split())
                        # Convert normalized dimensions to pixel dimensions
                        w_pixel = w * width
                        h_pixel = h * height

                        # Check if any detection is larger than the threshold
                        if w_pixel * h_pixel >= pixel_threshold:
                            # Copy both the txt and image file to the target directory
                            shutil.copy(filepath, os.path.join(target_dir, filename))
                            shutil.copy(image_path, os.path.join(target_dir, filename.replace('.txt', '.png')))
                            shutil.copy(cad_path, os.path.join(target_dir, filename.replace('Image.txt', 'Cad.png')))
                            break  # No need to check further detections for this file


# Example usage
source_directory = r'C:\work_space\data\rgb\aoi_ims_64\Export_26_05_24_32_42_0'
target_directory = r'C:\work_space\data\rgb\aoi_ims_64\Export_26_05_24_32_42_0_large'
copy_files_with_small_detections(source_directory, target_directory, pixel_threshold=25)