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

from multi_read_data import MemoryFriendlyLoader
import time


parser = argparse.ArgumentParser("SCI")
parser.add_argument('--data_path', type=str, default='C:/Users/kkzha/Downloads/track1.2_test_sample/track1.2_test_sample',
                    help='location of the data corpus')
parser.add_argument('--save_path', type=str, default='./results/DiffiDARKFACE', help='location of the data corpus')
parser.add_argument('--model', type=str, default='./weights/weights_771.pt', help='location of the data corpus')# ./EXP/Train-20240411-225725/model_epochs/weights_998.pt
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


def main():
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.cuda.set_device(args.gpu)

    cudnn.enabled = True
    cudnn.benchmark = True

    model = Finetunemodel(args.model)
    model = model.cuda()

    # Initialize running time
    total_time = 0.0
    total_images = 0

    model.eval()

    # Create CUDA events for time recording
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for _, (input, image_name) in enumerate(test_queue):
        with torch.no_grad():  # use torch.no_grad() instead of volatile, which is no longer supported in pytorch 2.x version
            total_images += 1
            input = input.cuda()  # Removed the Variable
            image_name = image_name[0].split('\\')[-1].split('.')[0]

            # Record start event
            start_event.record()

            i, r = model(input)
            end_event.record()
            u_name = '%s.png' % (image_name)
            print('processing {}'.format(u_name))
            u_path = save_path + '/' + u_name
            save_images(r, u_path)

            # Record end event
            end_event.record()

            # Wait for the end event to complete
            torch.cuda.synchronize()

            elapsed_time = start_event.elapsed_time(end_event)  # Unit is milliseconds

            print(f"GPU processing time for {u_name}: {elapsed_time}ms")

            total_time += elapsed_time

    print(f"Average GPU processing time per image: {total_time / total_images} ms")





if __name__ == '__main__':
    main()
