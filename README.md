# The environment I used:
cuda 11.8
numpy 1.24.3
opencv 4.6.0
python 3.8
pytorch 2.1.1

**This project is written using pycharm, so the directory are using / format**

# The file explanation:
## Folder Data prechecking
RawtoRGB.py transfers raw format images into BGR format using rawpy, other files are just using simple image resize and rename


## folder evaluation:
two files are for loss extraction from training loss .txt file, and plot.

# Root directory(Please Read this!! This part demonstrates what I've done)
- model.py containing the model details originated from the SCI author. For the reason of using 3 layers of 3x3 CNN, the SCI author conducted experiments regarding this topic in their paper, 3x3 is an optimal solution in performance and complexity.
- mult_read_data.py contains the data reading method for training and testing.
- finetune.py contains the fintune stratagy of SCI author, I did not conduct fintune in my training process.
- split.py is written by me, it is a random selection method using to divide different datasets.
- train.py contains the code to conduct model training. It was written by the SCI author, I made some changes to fit it to torch 2.x.
- test.py is the SCI author's code to test **one** specific model parameter, I added a time recording section and made changes to fit it to torch 2.x.
- test_2.py is a version of modified test.py. I added a loop to conduct test on multiple trained model parameters, conduct psnr test and take average for each weight. Then at the end of the loop findout the top 10 average psnr values.
- test_3.py is a version of modified test.py. It was used to generate figures using the designated weight.
- test_original.py is a version of an unmodified test.py, I only add a time recorder. When I first time run the model test, there were two major issues, one is float_tensor is no longer supported in torch 2.x, and the other one is that it used input = Variable() rather than input = input.to(device)
- the second issue does not affect the running of test.py, but it raised error. So I changed all Variable() to to.(device), and that caused a slower speed in running time. 
### So if you want to test on the running speed, you should use the test_original.py. 
- utils.py created by the SCI author. For the model size measurement, I used the function "def count_parameters_in_MB(model):" in this file, I also used this code into other models, EnGAN, Zero_DCE. Retinex Net has it's own function for this.


## Regarding the extra file I uploaded on moodle

### result.zip:
  - containing the original darkface_testset,
  - folder DarkFaceTest is the image selected for the face-detection test, models_vs_selected_img contains the selected darkface images and generated output from tested models.
  - weight758_MIT contains the test output of weight 758.

- I was planning on sending one of the entire training process of model training on my laptop, however, they are way too large that exceed the moodle limit. one in school server is 30GB and one on my laptop is 8GB with all te valset and weights.

- So instead, I upload the training log of both V1 and V3 trainset.
txtfile_TrainV1.zip
txtfile_TrainV3.zip
- Within them there is a top10.txt file, it contains the final top 10 average value of weights, and at top of each training txt file there is recorded the model size.
- I also attached the V2 traininglog.txt, which is the failed version that does not converge.

### dataset folder
contains the original LSRW dataset (Nikon), TestsetV1, raw format of TestsetV2 and trainset V1. Trainset V3 is too large

- I also attached the Ultra-Light-Fast-Generic-Face-Detector-1MB.zip, or you can find their github link https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
EnGAN https://github.com/VITA-Group/EnlightenGAN
Retinex Net https://github.com/weichen582/RetinexNet
Zero DCE https://github.com/Li-Chongyi/Zero-DCE
## Run env
- In which, EnGAN can be run on the same env as above mentioned with visdom  0.2.4 installed and referenced to this webpage https://blog.csdn.net/L_27N113E/article/details/130214102
- Retinex Net Zero_DCE and Face-detector all need to be run on their own env.
- The other models are all too large with size around 6-9G

# If you need any further details or datasets or need my version of other models or any training details, please contact my email with sid included in my report.
