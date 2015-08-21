import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from utils.cython_nms import nms
from utils.timer import Timer
import numpy as np
import caffe, os, cv2
import argparse
from test_utils import *

def gen_imgid(img_list):
    with open(img_list, 'r') as l:
        dic = {}
        for line in l:
            line = line.strip().split(' ')
            yield line[0], line[1]

def gen_result_str(id, label, dets):
    s = ''
    for det in dets:
        s = s + id + ' ' + str(label) + ' ' + str(det[-1]) + ' ' +\
            str(det[0]) + ' ' + str(det[1]) + ' ' + str(det[2]) + ' ' +\
            str(det[3]) + '\n'

    return s

def gen_time_str(id, prop_time, det_time, nms_time):
    return id + ' ' + str(prop_time) + ' ' + str(det_time) + ' ' +\
        str(nms_time) + ' ' +  str(prop_time + det_time + nms_time) + '\n'

if '__main__' == __name__:
    parser = argparse.ArgumentParser(description='Save the detection results')
    parser.add_argument('img_dir', help='Specify image path')
    parser.add_argument('img_list', help='Image ID list')
    parser.add_argument('--out', dest='out', help='Output file',
                        default='out.txt', type=str)
    parser.add_argument('--logtime', dest='logtime', help='Time performancei log',
                        default='time.txt', type=str)
    parser.add_argument('--confthr', dest='conf_thr', help='Threshold of confidence',
                        default=0.8, type=float)
    parser.add_argument('--nmsthr', dest='nms_thr', default=0.3, type=float,
                        help='Threshold for nms()')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='caffenet')
    parser.add_argument('--prop', dest='demo_prop', help='Method to generate proposals',
                        choices=PROP_GEN.keys(), default='sss')
    args = parser.parse_args()

    # Check image directory
    if not os.path.exists(args.img_dir):
        raise IOError('{:s} not found.' % args.Img_dir)

    # Create network
    net = get_net(args.net, args.cpu_mode, args.gpu_id)

    # Check proposal method
    prop, prop_opts = get_prop_gen(args.demo_prop)
    if prop is None:
        raise ValueError('Wrong proposal method.')

    f_out = open(args.out, 'w')
    f_time = open(args.logtime, 'w')
    for id1, id2 in gen_imgid(args.img_list):
        # Load image
        im = cv2.imread(os.path.join(args.img_dir, id1 + '.JPEG'))
        if im is None:
            continue

        # Compute object proposals
        pt = Timer()
        pt.tic()
        obj_proposals = prop.propose(im, *prop_opts)
        pt.toc()

        # Detect all object classes and regress object bounds
        dt = Timer()
        dt.tic()
        scores, boxes = im_detect(net, im, obj_proposals)
        dt.toc()

        # Run non-maximum suppression
        # Suppose label 0 is background, start from 1
        nt = Timer()
        for label in range(1, scores.shape[1]):
            nt.tic()
            cls_boxes = boxes[:, 4*label:4*(label + 1)]
            cls_scores = scores[:, label]
            dets = np.hstack((cls_boxes,
                            cls_scores[:, np.newaxis])).astype(np.float32)
            # Filter out low-confidence boxes
            dets = dets[np.where(dets[:, -1] >= args.conf_thr)]
            keep = nms(dets, args.nms_thr)
            dets = dets[keep, :]
            nt.toc()

            # Write detection results to file
            f_out.write(gen_result_str(id2, label, dets))

        f_time.write(gen_time_str(id2, pt.total_time, dt.total_time, nt.total_time))

    f_out.close()
    f_time.close()
