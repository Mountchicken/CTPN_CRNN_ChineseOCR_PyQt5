import sys
import os
import torch
import cv2
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QFileDialog
from Ui_weight import Ui_Form
from PIL import Image

from inference import *
class mywindow(QtWidgets.QWidget, Ui_Form):
    def __init__(self):
        super(mywindow, self).__init__()
        self.cwd=os.getcwd()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.load_image)
        self.pushButton_2.clicked.connect(self.predict)
        self.pushButton_3.clicked.connect(self.initialize)
        self.pushButton.setEnabled(False)
        self.pushButton_2.setEnabled(False)

    def load_image(self):
        img_path,filetype=QFileDialog.getOpenFileName(self,'open image',self.cwd,"*.JPG,*.JPEG,*.png,*.jpg,ALL Files(*)")
        if not img_path=='':
            self.img_path = img_path
            jpg = QtGui.QPixmap(img_path).scaled(self.label.width(), self.label.height())
            self.label.setPixmap(jpg)
    def predict(self):
        boxes, text_proposals, rh, rw = self.ctpn_model.detect(self.img_path) #boxes为大框，包含4个点坐标(6,4),text_proposals为小框,包含对角坐标点(94,4)
        for i, box in enumerate(boxes):
            box = box[:8].reshape(4, 2)
            box[:, 0] = box[:, 0] / rw #
            box[:, 1] = box[:, 1] / rh
            box = box.reshape(1, 8).astype(np.int32)
            box = [str(x) for x in box.reshape(-1).tolist()]

        #recognition text
        img = cv2.imread(self.img_path)
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
        for i, im in enumerate(croped_img):
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            converter = utils.strLabelConverter(self.config.DATASET.ALPHABETS)
            texts.append('{} :'.format(i) + recognition(self.config, im, self.crnn_model, converter, self.device) +'\n')

        #show image
        img_result = draw_img(self.img_path, boxes, text_proposals,texts,sorted_id)
        img_result = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
        new_h,new_w = self.get_img_scaled(img_result)
        img_result = cv2.resize(img_result,(new_w,new_h))
        img_result = np.asanyarray(img_result)
        img_result = QImage(img_result.data,img_result.shape[1],img_result.shape[0],img_result.shape[1]*img_result.shape[2],QImage.Format_RGB888)
        self.label.setPixmap(QPixmap(img_result))
        self.textBrowser.setText(''.join(texts))

    def initialize(self):
        config, args = parse_arg()
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.device = device
        #load crnn model
        print('loading crnn model')
        crnn_model = crnn.get_crnn(config).to(device)
        self.crnn_model = crnn_model
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
        self.ctpn_model = detect_obj
        self.textBrowser.setText('Finished')
        self.pushButton.setEnabled(True)
        self.pushButton_2.setEnabled(True)
        self.config = config
        self.args = args

    def get_img_scaled(self,img):
        label_width =self.label.width()
        label_heigth =self.label.height()
        max_size = max(label_width,label_heigth)
        img_max_size = max(img.shape[0],img.shape[1])
        scale = max_size/img_max_size
        return int(img.shape[0]*scale),int(img.shape[1]*scale)

if __name__=="__main__":
    app=QtWidgets.QApplication(sys.argv)
    myshow=mywindow()
    myshow.show()
    sys.exit(app.exec_())