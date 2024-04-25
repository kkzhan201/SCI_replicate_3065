import os
import sys
import numpy as np
import torch
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
from PIL import Image
from torch.autograd import Variable
from model import Finetunemodel
from torchvision import transforms
from multi_read_data import MemoryFriendlyLoader
import cv2

# This script serves to find the optimal psnr values among all 900 weights from 100-999, loop over all weights on the testsets, run the test.py content, and calculate average psnr.


parser = argparse.ArgumentParser("SCI")
parser.add_argument('--data_path', type=str, default='./data/psnrTest/lo3',#C:/Users/kkzha/Downloads/fivek_dataset/fivek_dataset/raw_photos/HQa1to700/testPNG, ./data/psnrTest/lo3
                    help='location of the data corpus')
parser.add_argument('--save_path', type=str, default='./results/weight211', help='location of the data corpus')
parser.add_argument('--model', type=str, default='./models/weights_219.pt', help='location of the data corpus')# ./EXP/Train-20240411-225725/model_epochs/weights_998.pt
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')

args = parser.parse_args()
save_path = args.save_path
os.makedirs(save_path, exist_ok=True)

TestDataset = MemoryFriendlyLoader(img_dir=args.data_path, task='test')

test_queue = torch.utils.data.DataLoader(
    TestDataset, batch_size=1,
    pin_memory=True, num_workers=0)


def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, 'png')


def main(args):
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    TestDataset = MemoryFriendlyLoader(img_dir=args.data_path, task='test')

    test_queue = torch.utils.data.DataLoader(TestDataset, batch_size=1, pin_memory=True, num_workers=0)
    print(f"Processing {len(test_queue)} images, saving to {args.save_path}")
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    model = Finetunemodel(args.model)
    model = model.cuda()

    model.eval()
    with torch.no_grad():
        for _, (input, image_name) in enumerate(test_queue):
            input = Variable(input, volatile=True).cuda()
            image_name = image_name[0].split('\\')[-1].split('.')[0]
            i, r = model(input)
            u_name = '%s.png' % (image_name)
            print('processing {}'.format(u_name))
            u_path = save_path + '/' + u_name
            save_images(r, u_path)
            print(f"Saved image {u_name} to {u_path}")

def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 10 * torch.log10(255.0 ** 2 / mse)

def psnr_cv2(img1, img2, max_val=255):
    img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10((max_val ** 2) / mse)

psnr_average_dict = {}
#P3065
if __name__ == '__main__':
    model_directory = 'E:/Programming_workplace/COMP3065/rob_New/model_epochs'#'./EXP/Train-20240411-225725/model_epochs'
    for weight_num in range(944, 1000):  # This line can be changed to 100, 1000 to loop over weight 100-999
        model_file = os.path.join(model_directory, f'weights_{weight_num}.pt')
        if os.path.isfile(model_file):
            args.model = model_file
            args.save_path = f'./results/ServerTrail1/weights_{weight_num}'
            print(f'Processing {args.model}')
            main(args)

            gt_folder = 'E:/Programming_workplace/COMP3065/Rob2/SCI/data/psnrTest/high2'#C:/Users/kkzha/Downloads/testdataTIF
#P3065
            mse = 0
            counter = 0
            psnr_file = open(f'./results/ServerTrail1/tstfile/weights_{weight_num}_psnr.txt', 'w')
            for image_name in os.listdir(args.save_path):
                gt_image_path = os.path.join(gt_folder, image_name)
                if os.path.isfile(gt_image_path):  # Check the ground truth
                    produced_image = cv2.imread(os.path.join(args.save_path, image_name))
                    gt_image = cv2.imread(gt_image_path)

                    # Convert images back to RGB
                    produced_image = cv2.cvtColor(produced_image, cv2.COLOR_BGR2RGB)
                    gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)

                    psnr_value = psnr_cv2(produced_image, gt_image)
                    mse += psnr_value
                    counter += 1

                    psnr_file.write(
                        f'{image_name}: {psnr_value}\n')
                else:
                    print(f"No ground truth found for {image_name} at {gt_image_path}")


            average_psnr = mse / counter if counter > 0 else 0
            psnr_file.write(
                f'Average PSNR for weights_{weight_num}: {average_psnr}')  # write the average PSNR to file

            # Add the weight number and average PSNR value to the dictionary
            psnr_average_dict[weight_num] = average_psnr

            psnr_file.close()

    # Listout the top 10 weights with the highest average PSNR
    top_10_weights = sorted(psnr_average_dict, key=psnr_average_dict.get, reverse=True)[:10]

    print("Top 10 weights with highest average PSNR:")
    for weight in top_10_weights:
        print(f'Weights_{weight}: {psnr_average_dict[weight]}')
