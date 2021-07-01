# CTPN_CRNN_ChineseOCR_PyQt5
CTPN and CRNN based Chinese OCR, developed with PyQt5

# Examples
- Hello guys, hope you are doing fine these days !😄
- In this repositories, i created a PyQt5 Application to do some Chinese OCR job which is based on CTPN and CRNN
- Here is the result, hope you enjoy it
- However, the CRNN model doesn't work well(Terrible at some situation😥). You will find it out when you try 
- ![test_example](https://github.com/Mountchicken/CTPN_CRNN_ChineseOCR_PyQt5/blob/main/github/detectd.JPG)

# Requirements
- PyQt5
- torch
- torchvision

# How to use it
- I'm sorry guys, i haven't find a way to deploy it, and you have to run it in your compiler 🙇‍♂️(VScode, pycharm or...)
## Build Enviroment
'''
cd CTPN_lib/bbox
python setup.py build
''''
## download pretrained weights
- The weights are larger than the uploading limit(25M below😅). Download them using BaiduYun
- Put them in CTPN_weights [CTPN weights(提取码:vqih)](https://pan.baidu.com/s/1OP4H87hunibVOQK_TKH-OA)
- Put them in CRNN_weights [CRNN weights(提取码:k4r4)](https://pan.baidu.com/s/1Ie-X_5Z-JuypKzsD3bRkzA)

## Choose which model to use
- In `inferrence.py`, from line 27 to line 32
- `argument: crnn_weights`: the file location of crnn weigth downloaded in the previous step
- `argument: ctpn_basemodel`:choose a ctpb backbone: vgg16, resnet50, shufflenet_v2_x1_0, mobilenet_v3_large, mobilenet_v3_small
- `argument: ctpn_weights`:corresponding ctpn weights with ctpn_base model downloaded in the previous step

## Run main_gui.py
- if you run the .py file succesfully, it should look like this
- ![test_example](https://github.com/Mountchicken/CTPN_CRNN_ChineseOCR_PyQt5/blob/main/github/menu.JPG)
- Then, you need to push the initialize button to load the model, after that, just wait the `Finished` sign appers in the right.
- ![test_example](https://github.com/Mountchicken/CTPN_CRNN_ChineseOCR_PyQt5/blob/main/github/Initialized.JPG)
- Finally, load the image with `Load Image` button and press `Detect`
- ![test_example](https://github.com/Mountchicken/CTPN_CRNN_ChineseOCR_PyQt5/blob/main/github/detectd.JPG)

# For more issue, contact me
- `Email Address` mountchicken@outlook.com
