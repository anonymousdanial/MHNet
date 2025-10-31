import os
import shutil
# Paths

# os.chdir("..")

source_folder = "dasatet/COD10k-v2/Train/Images/Image"  # change to your source folder
target_folder = "dasatet/COD10k-v2/Train/Images/camo_images"  # new folder to store COD10K-CAM images

def fuyo():
    if os.path.exists(target_folder):
        return"folder already exists"
    else:
        # Create target folder if it doesn't exist
        os.makedirs(target_folder, exist_ok=True)

        # Initialize counter
        count = 0

        # Iterate through files
        for filename in os.listdir(source_folder):
            if filename.startswith("COD10K-CAM"):
                source_path = os.path.join(source_folder, filename)
                target_path = os.path.join(target_folder, filename)
                shutil.copy2(source_path, target_path)  # copy file with metadata
                count += 1

        return f"Total COD10K-CAM images copied: {count}"