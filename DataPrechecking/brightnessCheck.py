import cv2
import numpy as np
import rawpy


# This script is for pre-selection images that has lower overall brightness

def get_image_brightness(image_path):
    with rawpy.imread(image_path) as raw:
        rgb = raw.postprocess()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    brightness = np.mean(gray)
    return brightness

image_path = "C:/Users/kkzha/Downloads/fivek_dataset/fivek_dataset/raw_photos/HQa1to700/low_light_photos/a0310-IMG_1898.dng"
brightness = get_image_brightness(image_path)

print(f"The brightness of the image is: {brightness}")
