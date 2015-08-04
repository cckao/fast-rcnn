import os.path as osp
import imp

thisDir = osp.dirname(__file__)

# Load LPO
imp.load_source('lpo', osp.join(thisDir, 'lpo', 'src', 'lpo.py'))
imp.load_source('BoundaryDetector', osp.join(thisDir, 'lpo', 'src', 'util.py'))
