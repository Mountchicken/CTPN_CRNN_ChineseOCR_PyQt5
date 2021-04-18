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

## download pretrained weights
- The weights are larger than the uploading limit(25M below😅). Download them using BaiduYun
- Put them in CTPN_weights [CTPN weights(提取码:vqih)](https://pan.baidu.com/s/1OP4H87hunibVOQK_TKH-OA)
- Put them in CRNN_weights [CRNN weights(提取码:k4r4)](https://pan.baidu.com/s/1Ie-X_5Z-JuypKzsD3bRkzA)

## Folders
- 'saved_vocab`: Contain serval vocabulary txt and you can also generate then during training
- `translation2019zh`: This is Google's chinese2english translation samples. It's huge and i only take the validation dataset to train

# How to use
## How to train
- 'Go inside the train.py, set some hyperparameters if you want or just run it!'
- 
## How to translate my own sentence
- `Go inside the inference.py, set the your own chinese sentence at line 73 

# Contact me for trained_weights(too big to upload)
- mountchicken@outlook.com
