import os

# This is a simple adding model names for identification of each photo script
# for example, just inserting '_SCI' before '.png', or it could be EnGAN, zeroDCE etc...

directory = 'E:/Programming_workplace/COMP3065/Rob2/SCI/results/DarkFaceTest/DF_Testingset'


for filename in os.listdir(directory):
    if filename.endswith('.png'):

        new_name = filename[:-4] + '_DF.png'
        original_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_name)
        os.rename(original_path, new_path)
        print(f"Renamed {filename} to {new_name}")