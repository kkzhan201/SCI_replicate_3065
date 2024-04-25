
import os
from PIL import Image

def convert_jpg_to_png(directory):
    for filename in os.listdir(directory):
        if os.path.splitext(filename)[1] == ".jpg":
            img = Image.open(f"{directory}/{filename}")
            img.save(f'{directory}/{os.path.splitext(filename)[0]}.png', "PNG")


convert_jpg_to_png("E:/Programming_workplace/COMP3065/Rob2/SCI/data/psnrTest/high2")