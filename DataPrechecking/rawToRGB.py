import cv2
import glob
import rawpy
import os
import shutil


def resize_image(image, size=512):
    h, w = image.shape[:2]
    if h > w:
        new_h, new_w = size * h / w, size
    else:
        new_h, new_w = size, size * w / h

    new_h, new_w = int(new_h), int(new_w)

    resized = cv2.resize(image, (new_w, new_h))
    return resized



image_folder_path = "C:/Users/kkzha/Downloads/fivek_dataset/fivek_dataset/raw_photos/HQa1to700/testDNG"
resized_folder_path = "C:/Users/kkzha/Downloads/fivek_dataset/fivek_dataset/raw_photos/HQa1to700/testPNG"

if not os.path.exists(resized_folder_path):
    os.makedirs(resized_folder_path)

image_paths = glob.glob(image_folder_path + "/*.dng")

for path in image_paths:

    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,  #Use camera white balance
            no_auto_bright=True,
            output_bps=8,       #Could be 16, but change to 8 to simulate real-life environment
            user_flip=0
        )

    resized_image = resize_image(rgb, size=512)

    bgr_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)

    filename = os.path.basename(path)
    output_path = os.path.join(resized_folder_path, filename.replace('.dng', '.png'))

    cv2.imwrite(output_path, bgr_image)
    print(f"Converted and saved: {output_path}")