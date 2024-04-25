import cv2
import os


def resize_images_in_folder(folder_path, target_height=480, target_width=640):
    """
    Resizes all images in the specified folder to the given dimensions.

    Parameters:
    - folder_path: str, path to the folder containing images.
    - target_height: int, desired image height.
    - target_width: int, desired image width.
    """
    # List all files in the folder
    files = os.listdir(folder_path)

    # Process each file in the folder
    for file in files:
        # Construct the full file path
        file_path = os.path.join(folder_path, file)

        # Check if the file is an image
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Read the image
            img = cv2.imread(file_path)

            # Check if the image was successfully opened
            if img is not None:
                # Resize the image
                resized_img = cv2.resize(img, (target_width, target_height))

                # Save the resized image, overwriting the original image
                cv2.imwrite(file_path, resized_img)
                print(f"Resized and saved {file}")


if __name__ == "__main__":
    # Specify the directory containing the images
    directory = 'E:/Programming_workplace/COMP3065/Rob2/SCI/results/DiffiDARKFACE'

    # Call the function
    resize_images_in_folder(directory)
