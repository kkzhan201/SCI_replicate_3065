import numpy as np
import cv2
import os


def compare_psnr(img1, img2, maxvalue):
    img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    return 10 * np.log10((maxvalue ** 2) / mse)



directory1 = 'E:/Programming_workplace/COMP3065/EnlightenGAN/EnlightenGAN/ablation/enlightening/fakeB/'
directory2 = './data/psnrTest/high/'




images1 = os.listdir(directory1)
images2 = os.listdir(directory2)

# To compare files with the same names within the folder, ground truth vs model output
common_images = list(set(images1) & set(images2))

total_psnr = 0
num_images = len(common_images)

for image_file in common_images:
    img1 = cv2.imread(directory1 + image_file)
    img2 = cv2.imread(directory2 + image_file)

    # Convert the images to RGB for psnr calculation
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)


    psnr = compare_psnr(img1, img2, 255)
    total_psnr += psnr

    print(f"PSNR for {image_file}: {psnr}")

# Average
average_psnr = total_psnr / num_images
print(f"Average PSNR: {average_psnr}")