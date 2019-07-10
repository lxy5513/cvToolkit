import cv2
import torch
import torchvision.transforms as transforms
import sys
import os
this_dir = os.path.split(os.path.realpath(__file__))[0]
lib_path = os.path.join(this_dir, 'lib')
sys.path.insert(0, lib_path)
import ipdb;pdb=ipdb.set_trace
from core.inference import get_final_preds
from utils.transforms import *
from config import cfg
from config import update_config
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

from object_detection.yolo_v3.human_detector import inference as yolo_infrence
from object_detection.yolo_v3.human_detector import load_model as yolo_model

class GetArgs():
    # hrnet config
    cfg = this_dir + '/lib/config/w32_256x192_adam_lr1e-3.yaml'
    modelDir= this_dir + '/models/pose_hrnet_w32_256x192.pth'
    dataDir=''
    logDir=''
    opts=[]
    prevModelDir=''

def upscale_bbox_fn(bbox, img, scale=1.25):
    new_bbox = []
    x0 = bbox[0]
    y0 = bbox[1]
    x1 = bbox[2]
    y1 = bbox[3]
    w = (x1 - x0) / 2
    h = (y1 - y0) / 2
    center = [x0 + w, y0 + h]
    new_x0 = max(center[0] - w * scale, 0)
    new_y0 = max(center[1] - h * scale, 0)
    new_x1 = min(center[0] + w * scale, img.shape[1])
    new_y1 = min(center[1] + h * scale, img.shape[0])
    new_bbox = [new_x0, new_y0, new_x1, new_y1]
    return new_bbox


def _box2cs(box, image_width, image_height):
    x, y, w, h = box[:4]
    return _xywh2cs(x, y, w, h, image_width, image_height)

def _xywh2cs(x, y, w, h, image_width, image_height):
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    aspect_ratio = image_width * 1.0 / image_height
    pixel_std = 200

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale

#### Pre-process
def pre_process(image, bboxs, scores, cfg, thred_score=0.8):

    if type(image) == str:
        data_numpy = cv2.imread(image, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    else:
        data_numpy = image

    inputs = []
    centers = []
    scales = []

    score_num = np.sum(scores>thred_score)
    max_box = min(5, score_num)
    for bbox in bboxs[:max_box]:
        x1,y1,x2,y2 = bbox
        box = [x1, y1, x2-x1, y2-y1]

        # 截取 box fron image  --> return center, scale
        c, s = _box2cs(box, data_numpy.shape[0], data_numpy.shape[1])
        centers.append(c)
        scales.append(s)
        r = 0

        trans = get_affine_transform(c, s, r, cfg.MODEL.IMAGE_SIZE)
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
            flags=cv2.INTER_LINEAR)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        input = transform(input).unsqueeze(0)
        inputs.append(input)

    inputs = torch.cat(inputs)
    return inputs, data_numpy, centers, scales


def load_model(config):
    sys.path.insert(0, this_dir)
    import hrnet_models
    model = hrnet_models.pose_hrnet.get_pose_net(
        config, is_train=False
    )
    model_file_name  = config.MODEL_DIR
    state_dict = torch.load(model_file_name)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k # remove module.
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    sys.path.remove(this_dir)
    return model

def get_pose_model():
    args = GetArgs()
    update_config(cfg, args)
    # load pose-hrnet MODEL
    pose_model = load_model(cfg)
    pose_model.cuda()
    return pose_model

def get_two_model():
    args = GetArgs()
    update_config(cfg, args)
    # load pose-hrnet MODEL
    pose_model = load_model(cfg)
    pose_model.cuda()
    # load YoloV3 Model
    bbox_model = yolo_model()
    return bbox_model, pose_model

def get_keypoints_from_bbox(pose_model, image, bbox):
    x1,y1,w,h = bbox
    bbox_input = []
    bbox_input.append([x1, y1, x1+w, y1+h])
    inputs, origin_img, center, scale = pre_process(image, bbox_input, scores=1, cfg=cfg)
    with torch.no_grad():
        # compute output heatmap
        inputs = inputs[:,[2,1,0]]
        output = pose_model(inputs.cuda())
        # compute coordinate
        preds, maxvals = get_final_preds(
            cfg, output.clone().cpu().numpy(), np.asarray(center), np.asarray(scale))

    # (N, 17, 3)
    result = np.concatenate((preds, maxvals), -1)
    return result

def get_keypoints(human_model, pose_model, image, smooth=None):
    bboxs, scores = yolo_infrence(image, human_model)
    # bbox is coordinate location
    inputs, origin_img, center, scale = pre_process(image, bboxs, scores, cfg)

    with torch.no_grad():
        # compute output heatmap
        inputs = inputs[:,[2,1,0]]
        output = pose_model(inputs.cuda())
        # compute coordinate
        preds, maxvals = get_final_preds(
            cfg, output.clone().cpu().numpy(), np.asarray(center), np.asarray(scale))

    # (N, 17, 3)
    result = np.concatenate((preds, maxvals), -1)
    return result


if __name__ == '__main__':
    bbox_model, pose_model = get_two_model()
    im_path = os.path.join(os.environ.get("CVTOOLBOX"), 'data/test.png')
    im = cv2.imread(im_path)
    result = get_keypoints(bbox_model, pose_model, im)
    from kps2d_detection.pose_utils import draw_img17
    draw_img17(im, result, 1, 2000)
