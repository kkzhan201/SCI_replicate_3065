import cv2
import numpy as np

# This is single file verification script on psnr

def compare_psnr(img1, img2, maxvalue):
    img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    return 10 * np.log10((maxvalue ** 2) / mse)



img1 = cv2.imread('./results/MediumNew2/a1299-NKIM_MG_6747.png')#E:/Programming_workplace/COMP3065/EnlightenGAN/EnlightenGAN/ablation/enlightening/test_200/images/1_fake_B.png
img2 = cv2.imread('C:/Users/kkzha/Downloads/a1299-NKIM_MG_6747.tif')#./data/psnrTest/high2/1.png
print(f"img1 shape: {img1.shape}")
print(f"img2 shape: {img2.shape}")

img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)
print(f"img2 shape: {img2.shape}")


psnr = compare_psnr(img1, img2, 255)
print(f'The PSNR of the two images is: {psnr}')
