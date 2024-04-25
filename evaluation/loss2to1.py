import matplotlib.pyplot as plt
import re

# This script is for plotting graphs based on loss recorded in training logs of two trails, such as V1 and V2, V1 and V3.

def get_losses(filename):
    epochs = []
    losses = []
    with open(filename, 'r') as file:
        for line in file:
            match = re.search('train 0*([0-9]+) ([0-9.]+)', line)
            if match is not None:
                epochs.append(int(match.group(1)))
                losses.append(float(match.group(2)))
    return epochs, losses


file1 = 'E:/Programming_workplace/COMP3065/rob_New/log.txt'
file2 = 'E:/Programming_workplace/COMP3065/Rob2/SCI/EXP/Train-20240411-225725/log.txt'

epochs1, losses1 = get_losses(file1)
epochs2, losses2 = get_losses(file2)

plt.plot(epochs1, losses1, color='blue', label='V3 dataset on GPU cluster')
plt.plot(epochs2, losses2, color='red', label='V1 dataset on 3080ti')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.grid(True)
plt.show()
