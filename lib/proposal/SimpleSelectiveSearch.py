import sys
import os
dlibPath= os.path.join(os.path.dirname(__file__), 'dlib')
print dlibPath
sys.path.append(dlibPath)
import dlib
import cv2
import time
import numpy

class SimpleSelectiveSearch(object):
    def __init__(self):
        print 'SimpleSelectiveSearch init'

    def propose(self, img, *options):
        rects = []
        startTime = time.time()
        tMinSize = options[0]
        dlib.find_candidate_object_locations(img, rects, min_size=tMinSize)
        endTime = time.time()
        print 'SimpleSelectiveSearch takes ' + str((endTime - startTime) * 1000) + 'ms'
        return self.list2NdArray(rects)

    @staticmethod
    def list2NdArray(rects):
	convertedRects = []
	for index, d in enumerate(rects):
		tempArr = [d.left(), d.top(), d.right(), d.bottom()]
		convertedRects.append(tempArr);
	rects_array = numpy.asarray(convertedRects)
        return rects_array

if False:
    from skimage import io
    import scipy.io

    test = SimpleSelectiveSearch()
    img = cv2.imread('/home/darrenl/000004.jpg', 1)
    print test.propose(img, *[300]).shape
