import sys
import os
import cv2
import numpy as np
import ipdb; pdb=ipdb.set_trace
import numpy as np
import asyncio

## Convert COCO keypoints to PoseTrack keypoints
def convert_coco_poseTrack(kps):
    """
        input:  N, 17, 3
            kp_names = [0:'nose', 1:'l_eye', 2:'r_eye', 3:'l_ear', 4:'r_ear', 5:'l_shoulder',
                        6:'r_shoulder', 7:'l_elbow', 8:'r_elbow', 9:'l_wrist', 10:'r_wrist',
                        11:'l_hip', 12:'r_hip', 13:'l_knee', 14:'r_knee', 15:'l_ankle', 16:'r_ankle']
        output: N, 15, 3
            {0: 'right_ankle', 1: 'right_knee', 2: 'right_hip', 3: 'left_hip', 4:'left_knee', 5:'left_ankle',  6:'right_wrist', 7:'right_elbow', 8:'right_shoulder',9:'left_shoulder', 10:'left_elbow', 11:'left_wrist',12:'head_bottom', 13:'nose', 14'head_top']

    """
    MAP ={16:0, 14:1, 12:2, 11:3, 13:4, 15:5, 10:6, 8:7, 6:8, 5:9, 7:10, 9:11, 0:13}

    pose_track_kps = np.zeros((kps.shape[0], 15, 3))

    for k in range(kps.shape[0]):
        pivel = (kps[k][5] + kps[k][6])/2
        head_bottom = (pivel+kps[k][0])/2 # 12
        head_top = 2*kps[k][0] - head_bottom # 14
        head_top[0] = (head_bottom[0] + kps[k][0][0])/2 #改变x

        for coco_i, pt_j in MAP.items():
            pose_track_kps[k][pt_j] = kps[k][coco_i]
        pose_track_kps[k][12] = head_bottom
        pose_track_kps[k][14] = head_top

    return pose_track_kps


async def async_get_keypoints_from_api(img, bbox, fast=None):
    if fast:
        #  url_pose='http://127.0.0.1:9094/predictions/simple_baseline'
        url_pose='http://127.0.0.1:9294/predictions/simple_baseline'
    else:
        url_pose='http://127.0.0.1:9092/predictions/hrnet'
    _, jepg = cv2.imencode('.jpg', img)
    file_pose = {'data': jepg.tobytes(), 'boxes': str([bbox])}
    re = await requests.post(url_pose, files=file_pose)
    kpts = eval(re.text)['result']
    return np.array(kpts)

async def async_inference_keypoints_api(test_data, raw_img, fast=0):
    image_id = test_data['img_id']
    category_id = 1
    bbox = test_data['bbox']
    coco_kps = await async_get_keypoints_from_api(raw_img, bbox, fast)
    keypoints = convert_coco_poseTrack(coco_kps)[0]
    keypoints[..., 2] = keypoints[..., 2] + 0.5
    score = np.mean(coco_kps[0][...,2], 0) + 0.5
    keypoints = keypoints.reshape(-1).tolist()
    keypoints = [round(i, 3) for i in keypoints]
    return keypoints, coco_kps[0].tolist()

def inference_keypoints_api(test_data, raw_img, fast):
    image_id = test_data['img_id']
    category_id = 1
    bbox = test_data['bbox']
    coco_kps = get_keypoints_from_api(raw_img, bbox, fast=1)
    keypoints = convert_coco_poseTrack(coco_kps)[0]
    keypoints[..., 2] = keypoints[..., 2] + 0.5
    score = np.mean(coco_kps[0][...,2], 0) + 0.5
    keypoints = keypoints.reshape(-1).tolist()
    keypoints = [round(i, 3) for i in keypoints]

    return keypoints, coco_kps[0].tolist()

def get_keypoints_from_api(img, bbox, fast=None):
    import requests
    if fast:
        url_pose='http://127.0.0.1:9294/predictions/simple_baseline'
    else:
        url_pose='http://127.0.0.1:9092/predictions/hrnet'
    _, jepg = cv2.imencode('.jpg', img)
    file_pose = {'data': jepg.tobytes(), 'boxes': str([bbox])}
    re = requests.post(url_pose, files=file_pose)
    kpts = eval(re.text)['result']
    return np.array(kpts)


def inference_keypoints_api_coco(test_data, raw_img):
    # 动作识别使用的,多返回coco本身关节点
    image_id = test_data['img_id']
    category_id = 1
    bbox = test_data['bbox']
    coco_kps = get_keypoints_from_api(raw_img, bbox, fast=1)
    keypoints = convert_coco_poseTrack(coco_kps)[0]
    keypoints[..., 2] = keypoints[..., 2] + 0.5
    score = np.mean(coco_kps[0][...,2], 0) + 0.5
    keypoints = keypoints.reshape(-1).tolist()
    keypoints = [round(i, 3) for i in keypoints]
    return keypoints, coco_kps[0].tolist()

# convert coco(17) to openpose_coco(18)
def convert17_18(coco_kps):
    # map:0, neck, 5, 7, 9, 6, 8, 10, 11, 13, 15, 12, 14, 16, 1, 2, 3, 4
    kpts = np.zeros(shape=(len(coco_kps), 18, 3))
    for i in range(len(coco_kps)):
        # neck point
        neck =  (coco_kps[i][5] + coco_kps[i][6])/2
        mapping = (coco_kps[i][0], neck, coco_kps[i][5], coco_kps[i][7], coco_kps[i][9], coco_kps[i][6], coco_kps[i][8], coco_kps[i][10], coco_kps[i][11], coco_kps[i][13], coco_kps[i][15], coco_kps[i][12], coco_kps[i][14], coco_kps[i][16], coco_kps[i][1], coco_kps[i][2], coco_kps[i][3], coco_kps[i][4])
        kpts[i] = np.concatenate(mapping, 0).reshape(-1, 3) #(18, 3)
    return kpts


