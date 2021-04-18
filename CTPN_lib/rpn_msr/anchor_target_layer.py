# -*- coding:utf-8 -*-
import numpy as np
import cv2
import numpy.random as npr
from CTPN_lib.bbox.bbox import bbox_overlaps

from CTPN_lib.bbox.bbox_transform import bbox_transform
from CTPN_lib.rpn_msr.config import Config as cfg
from CTPN_lib.rpn_msr.generate_anchors import generate_anchors
import pdb

def anchor_target_layer(image,rpn_cls_score, gt_boxes, im_info, _feat_stride=[16, ], anchor_scales=[16, ]):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    Parameters
    ----------
    rpn_cls_score: (1, H, W, Ax2) bg/fg scores of previous conv layer
    gt_boxes: (G, 5) vstack of [x1, y1, x2, y2, class]
    im_info: a list of [image_height, image_width, scale_ratios]
    _feat_stride: the downsampling ratio of feature map to the original input image
    anchor_scales: the scales to the basic_anchor (basic anchor is [16, 16])
    ----------
    Returns
    ----------
    rpn_labels : (HxWxA, 1), for each anchor, 0 denotes bg, 1 fg, -1 dontcare
    rpn_bbox_targets: (HxWxA, 4), distances of the anchors to the gt_boxes(may contains some transform)
                            that are the regression objectives
    rpn_bbox_inside_weights: (HxWxA, 4) weights of each boxes, mainly accepts hyper param in cfg
    rpn_bbox_outside_weights: (HxWxA, 4) used to balance the fg/bg,
                            beacuse the numbers of bgs and fgs mays significiantly different
    """
    try:
        for item in gt_boxes:
            color = (0,0,255)
            image1 = cv2.rectangle(image,(int(item[0]),int(item[1])),(int(item[2]),int(item[3])),color)
        cv2.imwrite('result_auchor.jpg',image1)
    except:
        print('warning!!!!!')


    _anchors = generate_anchors(scales=np.array(anchor_scales))  # 生成基本的anchor,一共10个
    _num_anchors = _anchors.shape[0]  # 10个anchor
    gt_boxes = np.array(gt_boxes)

    dontcareflag = gt_boxes[:,-1].reshape(-1) #取了标注的宽为16的gtbox的第六个维度（1代表有文字，-1代表文字模糊）
    gt_boxes = gt_boxes[:,:-1] #取了标注的宽为16的gtbox的前五个维度，第五个维度全是1，代表有物体

    # allow boxes to sit over the edge by a small amount
    _allowed_border = 0

    im_info = im_info[0]  # 图像的高宽及通道数

    assert rpn_cls_score.shape[0] == 1, 'Only single item batches are supported' #一次只处理一张图片

    # map of shape (..., H, W)
    height, width = rpn_cls_score.shape[1:3]  # feature-map的高宽 （66，66）

    # 1. Generate proposals from bbox deltas and shifted anchors #求出featuremap的角落点映射回原图时的位置（为了下一步选择pasitive objects
    shift_x = np.arange(0, width) * _feat_stride #shift_x[-1]=1040
    shift_y = np.arange(0, height) * _feat_stride #shift_y[-1]=1040
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)  # in W H order
    # K is H x W
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()  # 生成feature-map和真实image上anchor之间的偏移量
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = _num_anchors  # 10个anchor
    K = shifts.shape[0]  # 66*66，feature-map的宽乘高的大小
    all_anchors = (_anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))  # 相当于复制宽高的维度，然后相加
    all_anchors = all_anchors.reshape((K * A, 4)) #从feature_map映射回原图的所有anchors (66*66*10,5)
    total_anchors = int(K * A)

    # 仅保留那些还在图像内部的anchor，超出图像的都删掉
    inds_inside = np.where(
        (all_anchors[:, 0] >= -_allowed_border) &
        (all_anchors[:, 1] >= -_allowed_border) &
        (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
        (all_anchors[:, 3] < im_info[0] + _allowed_border)  # height
    )[0]

    # keep only inside anchors
    anchors = all_anchors[inds_inside, :]  # 保留那些在图像内的anchor (39996,61)
    
    dontcareflagAll = np.tile(dontcareflag,(anchors.shape[0],1))# shape (39996,61),相当于把原来的dontcareflag复制了39996行

    # 至此，anchor准备好了
    # --------------------------------------------------------------
    # label: 1 is positive, 0 is negative, -1 is dont care

    labels = np.empty((len(inds_inside),), dtype=np.float32) #shape (39996)
    labels.fill(-1)  # 初始化label，均为-1

    # 计算anchor和gt-box的overlap，用来给anchor上标签
    overlaps = bbox_overlaps(
        np.ascontiguousarray(anchors, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float))  # 假设anchors有x个，gt_boxes有y个，返回的是一个（x,y）的数组
    # 存放每一个anchor和每一个gtbox之间的overlap, overlaps.shape(39996, 61)

    argmax_overlaps = overlaps.argmax(axis=1)
    # 找到每个位置上10个anchor中与gtbox，overlap最大的那个,shape(39996,),
    # 相当于featuremap上生成的每个anchor都找到了再原图上与之overlap最大的GTbox（其实应该很多都为0）

    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps] #选出39996个anchor的匹配gt
    max_overlaps_dontcare = dontcareflagAll[np.arange(len(inds_inside)), argmax_overlaps] 
    #shape(39996,)39996个anchor匹配的gt有些是我们标注的字体模糊的gt，要被忽略（-1代表忽略）

    gt_argmax_overlaps = overlaps.argmax(axis=0)  # (A)#找到和每一个gtbox，overlap最大的那个anchor的index sjape(61,)
    gt_max_overlaps = overlaps[gt_argmax_overlaps,np.arange(overlaps.shape[1])] # (A)#找到和每一个gtbox，overlap最大的那个anchorsjape(61,)
    gt_argmax_overlaps_dontcare = dontcareflagAll[gt_argmax_overlaps,np.arange(overlaps.shape[1])] #同理，有些gt是要被忽略的

    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0] #shape[65,]
    gt_argmax_overlaps_dontcare = gt_argmax_overlaps_dontcare[ np.where(overlaps == gt_max_overlaps)[1]]

    if not cfg.RPN_CLOBBER_POSITIVES:
        # assign bg labels first so that positive labels can clobber them
        labels[max_overlaps < cfg.RPN_NEGATIVE_OVERLAP] = 0  # 先给背景上标签，小于0.3overlap的


    labels[gt_argmax_overlaps[gt_argmax_overlaps_dontcare==1]] = 1  # 每个位置上的10个anchor中overlap最大的认为是前景
    labels[(max_overlaps >= cfg.RPN_POSITIVE_OVERLAP) & (max_overlaps_dontcare==1)] = 1  # overlap大于0.7的认为是前景

    ###############################################
    index = np.where(labels==1)[0]
    fg_auchors = anchors[index]
    for item in fg_auchors:
        image = cv2.rectangle(image,(int(item[0]),int(item[1])),(int(item[2]),int(item[3])),(255,0,0))
    cv2.imwrite('result_fg.jpg',image) #把匹配上的anchor也画回原图
    #####################################################  


    if cfg.RPN_CLOBBER_POSITIVES:
        # assign bg labels last so that negative labels can clobber positives
        labels[max_overlaps < cfg.RPN_NEGATIVE_OVERLAP] = 0



    # 至此， 上好标签，开始计算rpn-box的真值
    # --------------------------------------------------------------
    bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])  # 根据anchor和gtbox计算得真值（anchor和gtbox之间的偏差，既真实的回归参数

    bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_inside_weights[labels == 1, :] = np.array(cfg.RPN_BBOX_INSIDE_WEIGHTS)  # 内部权重，前景就给1，其他是0

    bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    if cfg.RPN_POSITIVE_WEIGHT < 0:  # 暂时使用uniform 权重，也就是正样本是1，负样本是0
        positive_weights = np.ones((1, 4))
        negative_weights = np.zeros((1, 4))
    else:
        assert ((cfg.RPN_POSITIVE_WEIGHT > 0) &
                (cfg.RPN_POSITIVE_WEIGHT < 1))
        positive_weights = (cfg.RPN_POSITIVE_WEIGHT /
                            (np.sum(labels == 1)) + 1)
        negative_weights = ((1.0 - cfg.RPN_POSITIVE_WEIGHT) /
                            (np.sum(labels == 0)) + 1)
        
    bbox_outside_weights[labels == 1, :] = positive_weights  # 外部权重，前景是1，背景是0
    bbox_outside_weights[labels == 0, :] = negative_weights



    # map up to original set of anchors
    # 一开始是将超出图像范围的anchor直接丢掉的，现在在加回来
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)  # 这些anchor的label是-1，也即dontcare
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)  # 这些anchor的真值是0，也即没有值
    bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)  # 内部权重以0填充
    bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)  # 外部权重以0填充


    # labels
    labels = labels.reshape((1, height, width, A))  # reshap一下label
    rpn_labels = labels

    # bbox_targets
    bbox_targets = bbox_targets.reshape((1, height, width, A * 4))  # reshape
    rpn_bbox_targets = bbox_targets
    
    # bbox_inside_weights
    bbox_inside_weights = bbox_inside_weights.reshape((1, height, width, A * 4))
    rpn_bbox_inside_weights = bbox_inside_weights

    # bbox_outside_weights
    bbox_outside_weights = bbox_outside_weights.reshape((1, height, width, A * 4))
    rpn_bbox_outside_weights = bbox_outside_weights

    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights
    #shape (1,66,66,10), (1,66,66,40), (1,66,66,40), (1,66,66,40)
    #rpn_labels:feature map上每一个anchor对应回原图后是否匹配到物体,1为前景，0为背景，-1为dontcare(超出原图范围或者标注就是文字模糊)
    #rpn_bbox_targets:真实回归参数（回归4个参数）
    #rpn_bbox_inside_weights

def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
