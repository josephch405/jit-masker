# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2 import model_zoo
try:
    from detectron2.utils.logger import setup_logger
except ImportError:
    pass

# import some common libraries
import numpy as np
import cv2
import random
# import some common detectron2 utilities
try:
    from detectron2.utils.logger import setup_logger
except ImportError:
    pass
try:
    from detectron2.engine import DefaultPredictor
except ImportError:
    pass
try:
    from detectron2.config import get_cfg
except ImportError:
    pass

try:
    from detectron2.utils.visualizer import Visualizer
except ImportError:
    pass


try:
    from detectron2.data import MetadataCatalog
except ImportError:
    pass


mrcnn_50 = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
rcnn_101 = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"

class MODEL_ZOO():
    def __init__(self,model_name = "mrcnn_50"):
        setup_logger()
        self.cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not runn        ing a model in detectron2's core library
        if model_name == "mrcnn_50":
            config = mrcnn_50
        elif model_name == "rcnn_101":
            config = rcnn_101
        self.cfg.merge_from_file(model_zoo.get_config_file(config))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config)
        self.predictor = DefaultPredictor(self.cfg)

    def __call__(self, im):
        outputs = self.predictor(im)
        pred = outputs["instances"]
        result = pred.pred_masks[pred.pred_classes == 0,:,:].any(dim=0) > 0 
        return result,0,0,0,0,0,0
    
