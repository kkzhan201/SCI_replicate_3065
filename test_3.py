import os
import sys
import time
from matplotlib import pyplot as plt
import cv2
import numpy as np
import torch
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
from PIL import Image
from torch.autograd import Variable
from model import Finetunemodel

from multi_read_data import MemoryFriendlyLoader

parser = argparse.ArgumentParser("SCI")

parser.add_argument('--model', type=str, default='./weights/weights_999.pt', help='location of the data corpus')
parser.add_argument('--seed', type=int, default=2, help='random seed')

args = parser.parse_args()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Finetunemodel(args.model)
    model = model.to(device)

    model.eval()
    # dir_path = r'E:\Programming_workplace\COMP3065\Rob2\SCI\data\psnrTest\low'
    dir_path = r'C:\Users\kkzha\Downloads\track1.2_test_sample\track1.2_test_sample' #r'./data/medium'
    img_files = ['%s/%s' % (i[0].replace("\\", "/"), j) for i in os.walk(dir_path) for j in i[-1] if
                 j.endswith(('jpg', 'png', 'jpeg', 'JPG'))]

    for img_path in img_files:

        img_o = cv2.imread(img_path)

        img = img_o[:, :, ::-1].transpose(2, 0, 1).copy()

        start = time.time()

        img = torch.from_numpy(img).to(device)
        input = img.float()  
        input /= 255.0  # 0 - 255 to 0.0 - 1.0
        if input.ndimension() == 3:
            input = input.unsqueeze(0)

        with torch.no_grad():
            i, tensor = model(input)
        # u_name = '/%s.png' % (image_name)
        print(f'processing', img_path, 'time', time.time() - start)

        img_o = cv2.imread(img_path)
        img_o_rgb = cv2.cvtColor(img_o, cv2.COLOR_BGR2RGB)
        plt.subplot(1, 2, 1)
        # Run inference
        plt.imshow(img_o_rgb)
        plt.title('Original Image')
        image_numpy = tensor[0].cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)))


        result = np.clip(image_numpy * 255.0, 0, 255).astype('uint8')
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        plt.subplot(1, 2, 2)
        plt.imshow(result_rgb)
        plt.title('Result Image')
        cv2.imshow("result", cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.show()
        cv2.waitKey()


if __name__ == '__main__':
    main()