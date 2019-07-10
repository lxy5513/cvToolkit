import os
import sys
import pprint
import ipdb;pdb=ipdb.set_trace
import numpy as np
from tqdm import tqdm
import cv2

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

this_dir = os.path.split(os.path.realpath(__file__))[0]
lib_path = os.path.join(this_dir, 'lib')
sys.path.insert(0, lib_path)
from core.config import config
from core.config import update_config
from core.config import update_dir
from core.loss import JointsMSELoss
from core.function import validate
from utils.utils import create_logger
from utils.transforms import *
from core.inference import get_final_preds
import models as simple_baseline_models
sys.path.remove(lib_path)

from object_detection.yolo_v3.human_detector import inference as yolo_infrence
from object_detection.yolo_v3.human_detector import load_model as yolo_model

class GetArgs():
    cfg = this_dir + '/experiments/coco/resnet50/384x288_d256x3_adam_lr1e-3.yaml'
    model_file= this_dir + '/models/pytorch/pose_resnet_50_384x288.pth.tar'
    gpus=None
    workers=None
    use_detect_bbox=False
    post_process=False
    shift_heatmap=False
    coco_bbox_file=None
    det_file=''
    shift_heatmap=False
    flip_test=False
    frequent=100
    NUM_JOINTS=17


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

def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers
    if args.use_detect_bbox:
        config.TEST.USE_GT_BBOX = not args.use_detect_bbox
    args.flip_test = True
    if args.flip_test:
        config.TEST.FLIP_TEST = args.flip_test
    if args.post_process:
        config.TEST.POST_PROCESS = args.post_process
    if args.shift_heatmap:
        config.TEST.SHIFT_HEATMAP = args.shift_heatmap
    if args.model_file:
        config.TEST.MODEL_FILE = args.model_file
    if args.coco_bbox_file:
        config.TEST.COCO_BBOX_FILE = args.coco_bbox_file

###### Pre-process
def pre_process(data_numpy, bboxs, scores, cfg, thred_score=0.1):

    inputs = []
    centers = []
    scales = []

    score_num = np.sum(scores>thred_score)
    max_box = min(100, score_num)
    for bbox in bboxs[:max_box]:
        x1,y1,x2,y2 = bbox
        box = [x1, y1, x2-x1, y2-y1]

        # 截取 box from image  --> return center, scale
        c, s = _box2cs(box, data_numpy.shape[0], data_numpy.shape[1])
        centers.append(c)
        scales.append(s)
        r = 0

        trans = get_affine_transform(c, s, r, cfg.MODEL.IMAGE_SIZE)
        # 通过仿射变换截取人体图片
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


##### load model
def load_model(config):
    model = simple_baseline_models.pose_resnet.get_pose_net(
        config, is_train=False
    )

    state_dict = torch.load(config.TEST.MODEL_FILE)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k # remove module.
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    return model

def get_two_model():
    args = GetArgs()
    update_config(args.cfg)
    reset_config(config, args)

    # load pose-hrnet MODEL
    pose_model = load_model(config)
    pose_model.cuda().eval()

    # load YoloV3 Model
    bbox_model = yolo_model()
    return bbox_model, pose_model


def get_pose_model():
    args = GetArgs()
    update_config(args.cfg)
    reset_config(config, args)
    # load pose-hrnet MODEL
    pose_model = load_model(config)
    pose_model.cuda().eval()
    return pose_model

def get_keypoints_from_bbox(pose_model, image, bbox):
    x1,y1,w,h = bbox
    bbox_input = []
    bbox_input.append([x1, y1, x1+w, y1+h])
    inputs, origin_img, center, scale = pre_process(image, bbox_input, scores=1, cfg=config)
    with torch.no_grad():
        # compute output heatmap
        output = pose_model(inputs.cuda())
        # compute coordinate
        preds, maxvals = get_final_preds(
            config, output.clone().cpu().numpy(), np.asarray(center), np.asarray(scale))

    # (N, 17, 3)
    result = np.concatenate((preds, maxvals), -1)
    return result

def get_keypoints(human_model, pose_model, image):
    bboxs, scores = yolo_infrence(image, human_model)
    # bbox is coordinate location
    inputs, origin_img, center, scale = pre_process(image, bboxs, scores, config)

    with torch.no_grad():
        # compute output heatmap
        output = pose_model(inputs.cuda())
        # compute coordinate
        preds, maxvals = get_final_preds(
            config, output.clone().cpu().numpy(), np.asarray(center), np.asarray(scale))

    # (N, 17, 3)
    result = np.concatenate((preds, maxvals), -1)
    return result

def main():
    image_path = os.path.join(os.environ.get('CVTOOLBOX'), 'data/test.png')
    im = cv2.imread(image_path)
    bbox_model, pose_model = get_two_model()
    kps = get_keypoints(bbox_model, pose_model, im)
    from kps2d_detection.pose_utils import draw_img17
    draw_img17(im, kps, 1, 5000)

if __name__ == '__main__':
    main()
