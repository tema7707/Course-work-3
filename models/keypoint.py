import sys
sys.path.append('.')

import cv2
import numpy as np
import torch
import random


from detectron2_repo.projects.DensePose.densepose import add_densepose_config
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from .viton import make_image


class KeyPointPredictor:
    """
        Class that forecast human body keypoints

        Simple sequence to evaluate:

        1. Initialize this class with model weights and configuration.
        2. Set image to segmentate
        3. Get predictions like image in 2d np.array format

        If you'd like to do a simple prediction without anything more fancy, please refer to this example below,
        for more information, please, see https://github.com/facebookresearch/detectron2/tree/master/projects/DensePose

        Examples:

        .. code-block:: python

            dp = DensePosePredictor("detectron2_repo/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml", "./models/densepose_rcnn_R_50_FPN_s1x.pkl")
            image = cv2.imread("image2.jpg")  # predictor expects BGR image.
            head, body = dp.predict(image) # get masks
    """

    def __init__(self):
        self.cfg = self.setup_config()

    def setup_config(self):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml")
        cfg.freeze()
        return cfg

    def predict(self, img):
        predictor = DefaultPredictor(self.cfg)
        with torch.no_grad():
            outputs = predictor(img)["instances"].pred_keypoints.cpu().detach()[0]
            return KeyPointPredictor.format2viton(outputs)

    @staticmethod
    def format2viton(outputs):
        return make_image(outputs)