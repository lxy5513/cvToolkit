'''
使用yolov3作为pose net模型的前处理
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import ipdb;pdb=ipdb.set_trace
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from core.config import config
from core.config import update_config
from core.config import update_dir
from core.loss import JointsMSELoss
from core.function import validate
from utils.utils import create_logger
from utils.transforms import *
from lib.core.inference import get_final_preds
import cv2
#  import dataset
import models

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        #  required=True,
                        default='experiments/coco/resnet50/384x288_d256x3_adam_lr1e-3.yaml',
                        type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=config.PRINT_FREQ,
                        type=int)
    parser.add_argument('--gpus',
                        help='gpus',
                        type=str)
    parser.add_argument('--workers',
                        help='num of dataloader workers',
                        type=int)
    parser.add_argument('--model-file',
                        help='model state file',
                        default='models/pytorch/pose_resnet_50_384x288.pth.tar',
                        type=str)
    parser.add_argument('--use-detect-bbox',
                        help='use detect bbox',
                        action='store_true')
    parser.add_argument('--flip-test',
                        help='use flip test',
                        action='store_true')
    parser.add_argument('--post-process',
                        help='use post process',
                        action='store_true')
    parser.add_argument('--shift-heatmap',
                        help='shift heatmap',
                        action='store_true')
    parser.add_argument('--coco-bbox-file',
                        help='coco detection bbox file',
                        type=str)
    parser.add_argument('--det-file',
                        dest='det_file',
                        default='',
                        help='json file include image path and bbox',
                        type=str)

    args = parser.parse_args()

    return args
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
def PreProcess(image, bboxs, scores, cfg, thred_score=0.1):

    if type(image) == str:
        data_numpy = cv2.imread(image, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

    else:
        data_numpy = image

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


###### Pose-process
def PostProcess(image, preds, image_path):
    for i in range(len(preds)):
        for mat in preds[i]:
            # kpt
            x, y = int(mat[0]), int(mat[1])
            cv2.circle(image, (x, y), 2, (255, 0, 0), 2)

    name = 'output/'+os.path.basename(image_path)
    cv2.imwrite(name, image)
    img_path = os.path.join(os.getcwd(), name)
    print('image comes from ', image_path, '\nAfter handle ...')
    print('image save in ', img_path)
    #  cv2.imshow('pose',image)
    #  cv2.waitKey(3000)


##### load model
def model_load(config):
    model = eval('models.'+config.MODEL.NAME+'.get_pose_net')(
        config, is_train=False
    )
    state_dict = torch.load(config.TEST.MODEL_FILE)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k # remove module.
        print(name,'\t')
        new_state_dict[name] = v
    #  import ipdb;ipdb.set_trace()
    model.load_state_dict(new_state_dict)
    return model

def main():
    args = parse_args()
    reset_config(config, args)
    image_path = '/home/xyliu/Pictures/pose/soccer.png'

    ########## 加载human detecotor model
    #  from lib.detector.yolo.human_detector2 import load_model as yolo_model
    #  human_model = yolo_model()

    from lib.detector.mmdetection.high_api import load_model as mm_model
    human_model = mm_model()

    #  from lib.detector.yolo.human_detector2 import human_bbox_get as yolo_det
    from lib.detector.mmdetection.high_api import human_boxes_get as mm_det


    # load MODEL
    pose_model = model_load(config)
    pose_model.eval()
    pose_model.cuda()

    from pycocotools.coco import COCO
    annFile = '/ssd/xyliu/data/coco/annotations/instances_val2017.json'
    im_root = '/ssd/xyliu/data/coco/images/val2017/'
    coco = COCO(annFile)
    catIds = coco.getCatIds(catNms=['person'])
    # 所有人体图片的id
    imgIds = coco.getImgIds(catIds=catIds )
    kpts_result = []
    detected_image_num = 0
    box_num = 0
    for imgId in tqdm(imgIds):
        img = coco.loadImgs(imgId)[0]
        im_name = img['file_name']
        img = im_root + im_name
        img_input = cv2.imread(img, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        try:
            #  bboxs, scores = yolo_det(img_input, human_model)
            bboxs, scores = mm_det(human_model, img_input, 0.1)
            inputs, origin_img, center, scale = PreProcess(img_input, bboxs, scores, config)

        except Exception as e:
            print(e)
            continue

        detected_image_num += 1
        with torch.no_grad():
            output = pose_model(inputs.cuda())
            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), np.asarray(center), np.asarray(scale))

            #  vis = np.ones(shape=maxvals.shape,)
            vis = maxvals
            preds = preds.astype(np.float16)
            keypoints = np.concatenate((preds, vis), -1)
            for k, s in zip(keypoints, scores.tolist()):
                box_num += 1
                k = k.flatten().tolist()
                item = {"image_id": imgId, "category_id": 1, "keypoints": k, "score":s}
                kpts_result.append(item)


    num_joints = 17
    in_vis_thre = 0.2
    oks_thre = 0.5
    oks_nmsed_kpts = []
    for i in range(len(kpts_result)):
        img_kpts = kpts_result[i]['keypoints']
        kpt = np.array(img_kpts).reshape(17,3)
        box_score = kpts_result[i]['score']
        kpt_score = 0
        valid_num = 0
        # each joint for bbox
        for n_jt in range(0, num_joints):
            # score
            t_s = kpt[n_jt][2]
            if t_s > in_vis_thre:
                kpt_score = kpt_score + t_s
                valid_num = valid_num + 1
        if valid_num != 0:
            kpt_score = kpt_score / valid_num

        # rescoring 关节点的置信度 与 box的置信度的乘积
        kpts_result[i]['score'] = kpt_score * box_score


    import json
    data = json.dumps(kpts_result)
    print('image num is {} \tdetected_image num is {}\t person num is {}'.format(len(imgIds), detected_image_num, box_num)),
    #  data = json.dumps(str(kpts_result))
    with open('person_keypoints.json', 'wt') as f:
        #  pass
        f.write(data)

if __name__ == '__main__':
    main()
