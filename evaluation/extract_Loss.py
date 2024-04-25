import matplotlib.pyplot as plt
import re


epochs = []
losses = []

with open('E:/Programming_workplace/COMP3065/rob_New/log.txt', 'r') as file:
    for line in file:
        match = re.search('train 0*([0-9]+) ([0-9.]+)', line)
        if match is not None:
            epochs.append(int(match.group(1)))
            losses.append(float(match.group(2)))

plt.plot(epochs, losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.grid(True)
plt.show()
