import cv2
import numpy as np

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


class Segmentator:
    """
        Class that has one main goal - segmentate t-short

        Simple sequence to evaluate:

        1. Initialize this class with model configuration.
        2. Set image to segmentate
        3. Get predictions like image in 2d np.array format

        If you'd like to do a simple prediction without anything more fancy, please refer to this example below

        Examples:

        .. code-block:: python

            sgm = Segmentator()
            output = sgm.predict(input_img) # get 2d np.array
            cv2.imwrite("./output.jpg", output) # if you want to  save image
    """

    Model_R101 = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
    Model_50_1 = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    Model_50_3 = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"

    @staticmethod
    def grub_cut(image, outputs):
        instances = outputs.get('instances')
        classes = instances.pred_classes
        bboxes = instances.pred_boxes
        masks = instances.pred_masks

        # get first recognized t-shirt
        tshirt_mask = None
        for i in range(len(instances)):
            if classes[i] == 1:
                tshirt_mask = masks[i].cpu().numpy()
                region = bboxes[i].tensor.cpu().numpy()[0]
                break
        # we don't find any t-shirts
        if tshirt_mask is None:
            return np.zeros(image.shape)

        # some fake data
        background_model = np.zeros((1, 65), np.float64)
        foreground_model = np.zeros((1, 65), np.float64)
        # set marks for the grub cut algorithm
        mask = np.ones(image.shape[:2], np.uint8)
        mask[tshirt_mask == 1] = cv2.GC_PR_FGD
        mask[tshirt_mask != 1] = cv2.GC_PR_BGD
        # find center of the mask and set ground truth mark
        rect = (region[0], region[1], region[2], region[3])
        center_x, center_y = int((rect[1] + rect[3]) // 2), int((rect[0] + rect[2]) // 2)
        mask[center_x - 10:center_x + 10, center_y - 10:center_y + 10] = 1
        # set probably truth marks
        rect_mask = np.zeros(image.shape[:2], np.uint8)
        rect_mask[int(rect[1]):int(rect[3]), int(rect[0]):int(rect[2])] = 1

        mask, background_model, foreground_model = cv2.grabCut(image, mask, None, background_model, foreground_model, 5, cv2.GC_INIT_WITH_MASK)
        # keep only positive marks
        mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

        return image * mask[:, :, np.newaxis]

    def predict(self, image):
        '''
        Segmentate t-shirt from image. Cut all useless background and keep only a t-shirt from image.
        :param image: 2d np.array
        :return: 2d np.array
        '''
        outputs = self.predictor(image)
        return self.grub_cut(image, outputs)

    # TODO: Move models from our repo to a cloud storage
    def __init__(self, path_to_weights="./models/model_0024499.pth", model=Model_R101):
        # setup configuration of NN
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(model))
        cfg.DATALOADER.NUM_WORKERS = 4
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 45
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
        cfg.MODEL.WEIGHTS = path_to_weights
        self.predictor = DefaultPredictor(cfg)
