import sys

sys.path.append('..')

import io
import numpy as np
import cv2
import torch
import base64
from io import BytesIO
import PIL

from models.viton import Viton, make_image
from models.keypoint import KeyPointPredictor
from models.segmentation import Segmentator
from models.posenet import DensePosePredictor
from models.segmentation import Segmentator
from network_utils.network_utils import CPDataset
from torchvision import transforms

from flask import Flask, jsonify, request
from scipy.ndimage.filters import gaussian_filter
from PIL import Image

app = Flask(__name__)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform_1d = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


def pipeline(img, cloth, cloth_mask):
    img_array = np.array(img)
    cv2.imwrite("get.jpg", img_array[:, :, ::-1])
    image = cv2.imread("get.jpg")
    img = transform(img)
    dp = DensePosePredictor("../detectron2_repo/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml",
                            "../checkpoints_from_gd/densepose_rcnn_R_50_FPN_s1x.pkl")
    phead, body = dp.predict(image)
    # body = gaussian_filter(body, sigma=7)
    body = Image.fromarray((body * 255).astype(np.uint8))
    body = body.resize((24, 32), Image.BILINEAR)
    body = body.resize((img_array.shape[1], img_array.shape[0]), Image.BILINEAR)
    ########## BODY SHAPE ##########
    body = cv2.imread('/home/tema/Projects/Course-work-3/datasets/viton_resize/train/image-parse/000225_0.png',
                      cv2.IMREAD_GRAYSCALE)
    phead = np.zeros(body.shape)
    phead[body == 28] = 1
    phead[body == 75] = 1
    body[body == 28] = 0  # face
    body[body == 75] = 0  # hair
    body[body != 0] = 1
    body = Image.fromarray((body * 255).astype(np.uint8))
    body = body.resize((24, 32), Image.BILINEAR)
    body = body.resize((192, 256), Image.BILINEAR)
    ################################
    phead = torch.from_numpy(phead[np.newaxis, :, :]).type(torch.float32)
    head = img * phead - (1 - phead)
    shape = transform_1d(body).type(torch.float32)

    key_model = KeyPointPredictor()
    pose_map, im_pose = key_model.predict(img_array)

    viton = Viton(gmm_checkpoint_path='./../checkpoints_from_gd/gmm_final.pth',
                  tom_checkpoint_path='./../checkpoints_from_gd/tom_final.pth')
    res = viton.run_viton(head, pose_map, shape, cloth, cloth_mask)

    img_array[np.array(body) == -1] = 0
    cv2.imwrite("body.jpg", img_array)
    cv2.imwrite("body2.jpg", np.array(body))

    return encode((np.array(res[0].cpu().detach()).transpose(1, 2, 0) + 1) * 127.5)


def decode(byte64_str):
    bytes = BytesIO(base64.b64decode(byte64_str))
    pil_img = PIL.Image.open(bytes)
    return pil_img


def encode(img):
    cv2.imwrite('bb.jpg', img[:, :, ::-1])
    pil_img = Image.fromarray(img.astype('uint8'))
    pil_img = pil_img.convert('RGB')
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")


@app.route('/style/api/v1.0/segmentate', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.values['file']
        cloth_id = int(request.values['cloth'])
        if cloth_id == 0:
            c = Image.open(
                '/home/tema/Projects/Course-work-3/datasets/viton_resize/train/cloth/000736_1.jpg')  # path to cloth
            c = transform(c)
            cm = Image.open(
                '/home/tema/Projects/Course-work-3/datasets/viton_resize/train/cloth-mask/000736_1.jpg')  # path to cloth-mask
            cm_array = np.array(cm)
            cm_array = (cm_array >= 128).astype(np.float32)
            cm = torch.from_numpy(cm_array)
            cm.unsqueeze_(0)

        img = decode(file)
        res = pipeline(img, c, cm)
    return res


if __name__ == '__main__':
    app.run(debug=True, port=5000)
