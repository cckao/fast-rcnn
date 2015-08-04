import _lpoMods
import lpo.proposals as LpoProp
import lpo.segmentation as LpoSeg
import lpo.imgproc
from BoundaryDetector import getDetector
import cv2
import numpy

class LpoGenerator(object):
    def __init__(self, lpoModel, edgeDetType):
        self.lpoProp_ = LpoProp.LPO()
        self.lpoProp_.load(lpoModel)
        self.detector_ = getDetector(edgeDetType)

    def propose(self, img, *options):
        img1d = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img1d = img.ravel()
        return LpoGenerator.catPropsList(self.lpoProp_.propose(img1d.tolist(),
                                                  img.shape[1], img.shape[0],
                                                  img.shape[2], self.detector_,
                                                  *options))

    def setDetector(self, edgeDetType):
        self.detector_ = edgeDetType

    def setModel(self, lpoModel):
        self.lpoProp_.load(lpoModel)

    @staticmethod
    def catPropsList(propsList):
        rlt = propsList[0].toBoxes()
        for props in propsList[1:]:
            rlt = numpy.concatenate([rlt, props.toBoxes()])

        return rlt
