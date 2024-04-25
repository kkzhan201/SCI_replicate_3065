#"E:\Programming_workplace\COMP3065\Rob2\SCI\results\weight711_MIT\a1131-dvf_020.png"


import torch
import torchvision.transforms as transforms
from PIL import Image


def resize_image(image_path, output_size):

    image = Image.open(image_path)
    resize_transform = transforms.Resize(output_size)
    resized_image = resize_transform(image)
    resized_image.save('resized_image.png')



image_path = 'E:/Programming_workplace/COMP3065/Rob2/SCI/results/weight711_MIT/a1131-dvf_020.png'  # Path to your image
resize_image(image_path, (512, 340))