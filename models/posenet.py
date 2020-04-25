import cv2
import numpy as np
import torch
from detectron2_repo.projects.DensePose.densepose import add_densepose_config
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor


class DensePosePredictor:
    """
        Class that segmentate human body parts (head, body, arms, legs and e.g.)

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
    def __init__(self, cfg_path, model_path):
        self.cfg_path = cfg_path
        self.model_path = model_path

    def setup_config(self):
        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(self.cfg_path)
        cfg.MODEL.WEIGHTS = self.model_path
        cfg.freeze()
        return cfg

    def predict(self, img):
        cfg = self.setup_config()
        predictor = DefaultPredictor(cfg)
        with torch.no_grad():
            outputs = predictor(img)["instances"]
            return DensePosePredictor.execute_on_outputs(img, outputs)

    @staticmethod
    def execute_on_outputs(img, outputs):
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image = np.tile(image[:, :, np.newaxis], [1, 1, 3])

        if outputs is not None:
            densepose_output, bboxes_xywh = outputs.get('pred_densepose'), outputs.get('pred_boxes')
            S = densepose_output.S
            I = densepose_output.I
            N = S.size(0)

            for n in range(N):
                Sn = S[n].argmax(dim=0)
                In = I[n].argmax(dim=0) * (Sn > 0).long()
                matrix = In.cpu().numpy().astype(np.uint8)
                body = np.zeros(matrix.shape, dtype=np.uint8)
                head = np.zeros(matrix.shape, dtype=np.uint8)
                # generate masks of body and head
                body[matrix != 0] = 1
                head[matrix > 22] = 1

                # resize from 112*112 back to image size
                x_start, y_start, x_end, y_end = bboxes_xywh[n].tensor[0][:]
                x_start, y_start, x_end, y_end = int(x_start), int(y_start), int(x_end), int(y_end)
                head = cv2.resize(head, dsize=(x_end - x_start, y_end - y_start), interpolation=cv2.INTER_CUBIC)
                body = cv2.resize(body, dsize=(x_end - x_start, y_end - y_start), interpolation=cv2.INTER_CUBIC)
<<<<<<< HEAD

                head = np.concatenate((np.zeros((y_start, x_end - x_start)), head), axis=0)
                head = np.concatenate((head, np.zeros((max(image.shape[0] - y_end, 0), x_end - x_start))), axis=0)
                head = np.concatenate((np.zeros((image.shape[0], x_start)), head), axis=1)
                head = np.concatenate((head, np.zeros((image.shape[0], max(image.shape[1] - x_end, 0)))), axis=1)
                head = cv2.resize(head, dsize=(image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)

=======
                # body resize
>>>>>>> 83ba42a1350d845df5417beae79936a4b4aa9831
                body = np.concatenate((np.zeros((y_start, x_end - x_start)), body), axis=0)
                body = np.concatenate((body, np.zeros((max(image.shape[0] - y_end, 0), x_end - x_start))), axis=0)
                body = np.concatenate((np.zeros((image.shape[0], x_start)), body), axis=1)
                body = np.concatenate((body, np.zeros((image.shape[0], max(image.shape[1] - x_end, 0)))), axis=1)
                body = cv2.resize(body, dsize=(image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
<<<<<<< HEAD
                body[head == 1] = 0
=======
                # head resize
                head = np.concatenate((np.zeros((y_start, x_end - x_start)), head), axis=0)
                head = np.concatenate((head, np.zeros((max(image.shape[0] - y_end, 0), x_end - x_start))), axis=0)
                head = np.concatenate((np.zeros((image.shape[0], x_start)), head), axis=1)
                head = np.concatenate((head, np.zeros((image.shape[0], max(image.shape[1] - x_end, 0)))), axis=1)
                head = cv2.resize(head, dsize=(image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)

>>>>>>> 83ba42a1350d845df5417beae79936a4b4aa9831
                return head, body

        return None, None
