import os
import re
import numpy as np


# According to the MIT dataset standard, in a txt file it saved image names with respect to the environment taken.
# this script is try to select the night
def extract_numbers(directory):
    numbers = []
    pattern = re.compile(r'a(\d+)-')  # Pattern to match 'a' followed by numbers and a dash

    # Loop through each file in the directory
    for filename in os.listdir(directory):
        match = pattern.search(filename)
        if match:
            number = int(match.group(1))
            numbers.append(number)

    # Sort numbers in ascending order
    numbers_sorted = np.sort(numbers)

    # Write sorted numbers to a text file
    with open('sorted_numbers.txt', 'w') as file:
        for number in numbers_sorted:
            file.write(f'{number}\n')

    print(f'Sorted numbers are written to sorted_numbers.txt')

# Example usage
directory_path = 'C:/Users/kkzha/Downloads/fivek_dataset/fivek_dataset/raw_photos/HQa1to700/testPNG'
extract_numbers(directory_path)