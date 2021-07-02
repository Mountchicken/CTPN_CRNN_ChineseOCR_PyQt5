import numpy as np
import time
import cv2
import torch
from torch.autograd import Variable
import CRNN_lib.utils.utils as utils
import models.crnn as crnn
import CRNN_lib.config.alphabets as alphabets
import yaml
from easydict import EasyDict as edict
import argparse

import cv2
import numpy as np
import torch.nn.functional as F
from CTPN_lib.rpn_msr.proposal_layer import proposal_layer
from CTPN_lib.text_connector.detectors import TextDetector
from torchvision.transforms import transforms
from models.ctpn import *
import time
from PIL import Image, ImageDraw, ImageFont


def parse_arg():
    parser = argparse.ArgumentParser(description="demo")
    parser.add_argument('--cfg', help='experiment configuration filename', type=str, default='CRNN_lib/config/360CC_config.yaml')
    parser.add_argument('--img_path', type=str, default='images/Image10.jpg', help='the path to your image')
    parser.add_argument('--crnn_weights', type=str, default='CRNN_weights/mixed_second_finetune_acc_97P7.pth',
                        help='the path to your crnn weights')
    parser.add_argument('--ctpn_weights', type=str, default='CTPN_weights/resnet50.pth',
                        help='the path to your crnn weights')
    parser.add_argument('--ctpn_basemodel', type=str, default='resnet50', help='the path to your image')
    parser.add_argument('--show_image', type=bool, default=True, help='show the labeled image')
   
    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
        config = edict(config)

    config.DATASET.ALPHABETS = alphabets.alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    return config, args

def recognition(config, img, model, converter, device):

    h, w = img.shape
    # fisrt step: resize the height and width of image to (32, x)
    img = cv2.resize(img, (0, 0), fx=config.MODEL.IMAGE_SIZE.H / h, fy=config.MODEL.IMAGE_SIZE.H / h, interpolation=cv2.INTER_CUBIC)

    # second step: keep the ratio of image's text same with training
    h, w = img.shape
    w_cur = int(img.shape[1] / (config.MODEL.IMAGE_SIZE.OW / config.MODEL.IMAGE_SIZE.W))
    img = cv2.resize(img, (0, 0), fx=w_cur / w, fy=1.0, interpolation=cv2.INTER_CUBIC)
    img = np.reshape(img, (config.MODEL.IMAGE_SIZE.H, w_cur, 1))

    # normalize
    img = img.astype(np.float32)
    img = (img / 255. - config.DATASET.MEAN) / config.DATASET.STD
    img = img.transpose([2, 0, 1])
    img = torch.from_numpy(img)

    img = img.to(device)
    img = img.view(1, *img.size())
    model.eval()
    preds = model(img)
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    return sim_pred

def resize_image(img,max_size=1200,color=(0,0,0)):

    img_size = img.shape
    im_size_max = np.max(img_size[0:2])
    im_scale = float(max_size) / float(im_size_max)
    new_h = int(img_size[0] * im_scale)
    new_w = int(img_size[1] * im_scale)

    new_h_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
    new_w_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16

    re_im = cv2.resize(img, (new_w_w, new_h_h), interpolation=cv2.INTER_LINEAR)

    return re_im, (im_scale*(new_h_h/new_h),im_scale*(new_w_w/new_w))


def toTensorImage(image, is_cuda=True):
    image = transforms.ToTensor()(image)
#     image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image).unsqueeze(0)
    image = (image).unsqueeze(0)
    if (is_cuda is True):
        image = image.cuda()
    return image


class DetectImg():

    def load_model(self, model_file,base_model,detect_type,device):
        model_dict = torch.load(model_file)
        model = CTPN_Model(base_model,pretrained=False).to(device)
        model.load_state_dict(model_dict)
        self.model = model
        self.detect_type = detect_type
        self.model.eval()

    def detect(self, img_file):
        img = Image.open(img_file).convert('RGB')
        img = np.array(img)
        img_ori, (rh, rw) = resize_image(img) #img_ori为resize后图片，rh，rw为高宽的缩放系数
        h, w, c = img_ori.shape
        im_info = np.array([h, w, c]).reshape([1, 3]) #保存resize后图片的尺寸信息
        img = toTensorImage(img_ori)
        with torch.no_grad():
            pre_score, pre_reg, refine_ment = self.model(img) #模型推理
        score = pre_score.reshape((pre_score.shape[0], 10, 2, pre_score.shape[2], pre_score.shape[3])).squeeze(
            0).permute(0, 2, 3, 1).reshape((-1, 2)) #reshape ---> [55480,2]
        score = F.softmax(score, dim=1)
        score = score.reshape((10, pre_reg.shape[2], -1, 2)) #reshape --->[10,76,73,2]

        pre_score = score.permute(1, 2, 0, 3).reshape(pre_reg.shape[2], pre_reg.shape[3], -1).unsqueeze(
            0).cpu().detach().numpy() #reshape --->[1,76,73,20]
        pre_reg = pre_reg.permute(0, 2, 3, 1).cpu().detach().numpy() #reshape --->(1,76,73,40)
        refine_ment = refine_ment.permute(0, 2, 3, 1).cpu().detach().numpy() #reshape --->(1,76,73,10)

        textsegs, _ = proposal_layer(pre_score, pre_reg, refine_ment, im_info) #textsegs:候选框位置以及score信息 (500,5)
        scores = textsegs[:, 0] #分数
        textsegs = textsegs[:, 1:5] #对角坐标点位置

        textdetector = TextDetector(DETECT_MODE = self.detect_type)
        boxes, text_proposals = textdetector.detect(textsegs, scores[:, np.newaxis], img_ori.shape[:2]) #还要把score较低的proposal过滤掉
        boxes = np.array(boxes, dtype=np.int32)
        text_proposals = text_proposals.astype(np.int32)
        return boxes, text_proposals, rh, rw


def draw_img(im_file, boxes, text_proposals,texts,sorted_id):
    img_ori = cv2.imread(im_file)
    img_ori, (rh, rw) = resize_image(img_ori)
    for item in text_proposals: #把宽度为16的小框先画上变形图
        img_ori = cv2.rectangle(img_ori, (item[0], item[1]), (item[2], item[3]), (0, 255, 255))
    img_ori = cv2.resize(img_ori, None, None, fx=1.0 / rw, fy=1.0 / rh, interpolation=cv2.INTER_LINEAR)
    for i, box in enumerate(boxes):#把大框画回原图
        cv2.polylines(img_ori, [box[:8].astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0),
                      thickness =2)
        #显示识别文字
        if (isinstance(img_ori, np.ndarray)):  # 判断是否OpenCV图片类型
            img = Image.fromarray(cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB))
        # 创建一个可以在给定图像上绘图的对象
        draw = ImageDraw.Draw(img)
        # 字体的格式
        fontStyle = ImageFont.truetype("font/simsun.ttc", size=int((box[7]-box[1])*0.7), encoding="utf-8")
        # 绘制文本
        draw.text((box[0], box[1]-(box[7]-box[1])), str(sorted_id.index(i))+' '+texts[sorted_id.index(i)], fill ='red', font=fontStyle)
        # 转换回OpenCV格式
        img_ori =  cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    img_ori = cv2.resize(img_ori, None, None, fx=0.9, fy=0.9, interpolation=cv2.INTER_LINEAR)
    return img_ori

if __name__ == '__main__':

    config, args = parse_arg()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    #load crnn model
    print('loding crnn model')
    crnn_model = crnn.get_crnn(config).to(device)
    checkpoint = torch.load(args.crnn_weights)
    if 'state_dict' in checkpoint.keys():
        crnn_model.load_state_dict(checkpoint['state_dict'])
    else:
        crnn_model.load_state_dict(checkpoint)

    #load ctpn model
    print('loding ctpn model')
    detect_type = 'H'
    detect_obj = DetectImg()
    detect_obj.load_model(args.ctpn_weights,args.ctpn_basemodel,detect_type,device=device)

    #detect text
    print('Detecting text')
    boxes, text_proposals, rh, rw = detect_obj.detect(args.img_path) #boxes为大框，包含4个点坐标(6,4),text_proposals为小框,包含对角坐标点(94,4)

    for i, box in enumerate(boxes):
            box = box[:8].reshape(4, 2)
            box[:, 0] = box[:, 0] / rw #
            box[:, 1] = box[:, 1] / rh
            box = box.reshape(1, 8).astype(np.int32)
            box = [str(x) for x in box.reshape(-1).tolist()]

    #recognition text
    img = cv2.imread(args.img_path)
    croped_img = []
    ymins = []
    for i in range(boxes.shape[0]):
        box = boxes[i]#(8,1)-->(8,)
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
        x3 = box[4]
        y3 = box[5]
        x4 = box[6]
        y4 = box[7]
        xmin = min(x1,x2,x3,x4)
        xmax = max(x1,x2,x3,x4)
        ymin = min(y1,y2,y3,y4)
        ymax = max(y1,y2,y3,y4)
        ymins.append(ymin)
        croped_img.append(img[ymin:ymax,xmin:xmax])
    sorted_id = sorted(range(len(ymins)), key=lambda k: ymins[k], reverse=False)
    temp=[]
    for i in range(len(ymins)):
        temp.append(croped_img[sorted_id[i]])
    croped_img = temp

    texts = []
    for im in croped_img:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        converter = utils.strLabelConverter(config.DATASET.ALPHABETS)
        texts.append(recognition(config, im, crnn_model, converter, device))

    #show image
    img_result = draw_img(args.img_path, boxes, text_proposals,texts,sorted_id)
    for i, text in enumerate(texts):
        print('{} : {}'.format(i, text))
    if args.show_image:
        cv2.namedWindow('image',cv2.WINDOW_NORMAL)
        cv2.imshow('image',img_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
