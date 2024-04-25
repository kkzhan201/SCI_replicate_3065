import os
import shutil
import random


def move_random_files(source_directory, destination_directory, file_extension, num_files):
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    all_files = [f for f in os.listdir(source_directory) if f.endswith(file_extension)]
    rand_files = random.sample(all_files, num_files)

    for file_name in rand_files:
        shutil.move(os.path.join(source_directory, file_name), destination_directory)


move_random_files('./trainA', 'newlow', '.png', 430)
