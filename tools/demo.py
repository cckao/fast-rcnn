#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from utils.cython_nms import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
from test_utils import *

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

def demo(net, image_name, classes, prop=None, prop_opts=[], im_file=None):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    # By default, lead the image under data/demo/[image_name]
    if image_name:
        im_file = os.path.join(cfg.ROOT_DIR, 'data', 'demo', image_name + '.jpg')

    if im_file is None:
        raise 'Cannot find the image to detect'
    im = cv2.imread(im_file)

    # Load pre-computed Selected Search object proposals
    propTimer = Timer()
    propTimer.tic()
    if prop is None:
        box_file = os.path.join(cfg.ROOT_DIR, 'data', 'demo',
                                image_name + '_boxes.mat')
        obj_proposals = sio.loadmat(box_file)['boxes']
    else:
        obj_proposals = prop.propose(im, *prop_opts)
    propTimer.toc()
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im, obj_proposals)
    timer.toc()
    print '\tProposal method : ' + str(type(prop))
    print ('\tFinding proposals took {:.3f}s for '
            '{:d} object proposals').format(propTimer.total_time, obj_proposals.shape[0])
    print ('\tObjects Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls in classes:
        cls_ind = CLASSES.index(cls)
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        print 'All {} detections with p({} | box) >= {:.1f}'.format(cls, cls,
                                                                    CONF_THRESH)
        vis_detections(im, cls, dets, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--prop', dest='demo_prop', help='Method to generate proposals',
                        choices=PROP_GEN.keys(), default='pre')
    parser.add_argument('--img', dest='img_path', help='Specify image path', default=None)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    net = get_net(args.demo_net, args.cpu_mode, args.gpu_id)

    prop, prop_opts = get_prop_gen(args.demo_prop)

    specified_img_path = args.img_path

    #  Detect the specified image
    if specified_img_path and prop:
        demo(net, None, ('car',), prop, prop_opts, im_file=specified_img_path)
    else:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/000004.jpg'
        demo(net, '000004', ('car',), prop, prop_opts)

        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/001551.jpg'
        demo(net, '001551', ('sofa', 'tvmonitor'), prop, prop_opts)

    plt.show()
