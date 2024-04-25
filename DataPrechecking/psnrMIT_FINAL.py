import numpy as np
import cv2
import os


# This script is conducting batch average psnr calculation on model output with ground truth values
def compare_psnr(img1, img2, maxvalue):

    img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')  #Identical imgs should have infinite PSNR
    return 10 * np.log10((maxvalue ** 2) / mse)

def load_and_resize(img_path, shape):

    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR to RGB
    if img.shape != shape:
        img = cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    return img


directory1 = 'E:/Programming_workplace/COMP3065/Rob2/SCI/results/weight758_MIT'#weight169_MIT
directory2 = 'C:/Users/kkzha/Downloads/testdataTIF'

images1 = {os.path.splitext(f)[0]: f for f in os.listdir(directory1)}
images2 = {os.path.splitext(f)[0]: f for f in os.listdir(directory2)}

# need to find common image names in both folders
common_image_bases = set(images1.keys()).intersection(images2.keys())

total_psnr = 0
num_images = len(common_image_bases)

if num_images == 0:
    print("No common images to compare.")
else:
    for image_base in common_image_bases: #common image pairs
        img1_path = os.path.join(directory1, images1[image_base])
        img2_path = os.path.join(directory2, images2[image_base])

        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        if img1 is None or img2 is None:
            print(f"Error reading images for {image_base}. Skipping...")
            continue

        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)

        psnr = compare_psnr(img1, img2, 255)
        total_psnr += psnr
        print(f"PSNR for {image_base}: {psnr}")

    # Calculate the average PSNR
    average_psnr = total_psnr / num_images
    print(f"Average PSNR: {average_psnr}")