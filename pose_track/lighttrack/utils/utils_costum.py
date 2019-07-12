import numpy as np
import cv2
from cv_utils import add_text
import ipdb; pdb=ipdb.set_trace

## Convert COCO keypoints to PoseTrack keypoints
def conver_coco_poseTrack(kps):
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

def draw_bbox(img, bbox, text=None, change_color=None):
    x1, y1, w, h = bbox
    pt1 = (int(x1), int(y1))
    pt2 = (int(x1+w), int(y1+h))
    if change_color:
        img = cv2.rectangle(img, pt1, pt2, (2,0,2), 3, 2)
    else:
        img = cv2.rectangle(img, pt1, pt2, (255,0,255), 3, 2)

    if text:
        img = add_text(img, str(text), pt1, 2)
    return img

