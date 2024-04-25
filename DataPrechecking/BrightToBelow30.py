import cv2
import glob
import numpy as np
import rawpy
import shutil
import os
# This script is for pre-selection images that has lower overall brightness

def is_low_light(image_path, brightness_threshold=30):
    with rawpy.imread(image_path) as raw:
        rgb = raw.postprocess()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    brightness = np.mean(gray)
    return brightness < brightness_threshold


image_folder_path = "C:/Users/kkzha/Downloads/fivek_dataset/fivek_dataset/raw_photos/HQa1to700/low_light_photos"
low_light_folder_path = "C:/Users/kkzha/Downloads/fivek_dataset/fivek_dataset/raw_photos/HQa1to700/low_30"
image_paths = glob.glob(image_folder_path + "/*.dng")

for path in image_paths:
    print(f"Checking image: {path}")
    if is_low_light(path):
        filename = os.path.basename(path)
        full_low_light_path = os.path.join(low_light_folder_path, filename)
        shutil.move(path, full_low_light_path)
        print(f"Moved: {path}")

low_light_images = glob.glob(low_light_folder_path + "/*.dng")
print("Moved {} low-light images.".format(len(low_light_images)))
