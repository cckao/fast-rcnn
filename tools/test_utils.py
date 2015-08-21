import _init_paths
from fast_rcnn.config import cfg
import caffe, os
import matplotlib.pyplot as plt
import numpy as np
import LpoProposal
import SimpleSelectiveSearch

NETS = {'vgg16': ('VGG16',
                  'vgg16_fast_rcnn_iter_40000.caffemodel'),
        'vgg_cnn_m_1024': ('VGG_CNN_M_1024',
                           'vgg_cnn_m_1024_fast_rcnn_iter_40000.caffemodel'),
        'caffenet': ('CaffeNet',
                     'caffenet_fast_rcnn_iter_40000.caffemodel')}

PROP_GEN = {'pre': {},
            'lpo': {'model_path': '../lib/proposal/lpo/models/lpo_VOC_0.03.dat',
                    'b_det': 'mssf',
                    'appr_n': 1000},
            'sss': {'min_size': 300}}

def get_net(net_name, cpu_mode, gpu_id):
    prototxt = os.path.join(cfg.ROOT_DIR, 'models', NETS[net_name][0],
                            'test.prototxt')
    caffemodel = os.path.join(cfg.ROOT_DIR, 'data', 'fast_rcnn_models',
                              NETS[net_name][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_fast_rcnn_models.sh?').format(caffemodel))

    if cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)
    return net

def get_prop_gen(name):
    args = PROP_GEN[name]
    if name == 'lpo':
        model_path = os.path.join(os.path.dirname(__file__),
                                  args['model_path'])
        prop = LpoProposal.LpoGenerator(model_path, args['b_det'])
        return prop, [args['appr_n']]
    elif name == 'sss':
        return SimpleSelectiveSearch.SimpleSelectiveSearch(), [args['min_size']]
    else:
        return None, []

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

