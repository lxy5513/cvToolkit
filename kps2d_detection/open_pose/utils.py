import cv2
import numpy as np
import os

class common():
    keypoints_symmetry = [[1, 3, 5, 7, 9, 11, 13, 15],[2, 4, 6, 8, 10, 12, 14, 16]]
    rot = np.array([ 0.14070565, -0.15007018, -0.7552408 ,  0.62232804], dtype=np.float32)
    skeleton_parents =  np.array([-1,  0,  1,  2,  0,  4,  5,  0,  7,  8,  9,  8, 11, 12,  8, 14, 15])
    pairs = [(1,2), (5,4),(6,5),(8,7),(8,9),(10,1),(11,10),(12,11),(13,1),(14,13),(15,14),(16,2),(16,3),(16,4),(16,7)]

    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list([4, 5, 6, 11, 12, 13]), list([1, 2, 3, 14, 15, 16])
    pad = (243 - 1) // 2 # Padding on each side
    causal_shift = 0
    joint_pairs = [[0, 1], [1, 3], [0, 2], [2, 4],
                [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
                [5, 11], [6, 12], [11, 12],
                [11, 13], [12, 14], [13, 15], [14, 16]]
    openpose_joint_pairs = [[0, 1],[0, 14], [0, 15], [14, 16], [15, 17], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7]
            , [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13]]


def resize_img(frame, max_length=640):
    H, W = frame.shape[:2]
    if max(W, H) > max_length:
        if W>H:
            W_resize = max_length
            H_resize = int(H * max_length / W)
        else:
            H_resize = max_length
            W_resize = int(W * max_length / H)
        frame = cv2.resize(frame, (W_resize, H_resize), interpolation=cv2.INTER_AREA)
        return frame, W_resize, H_resize

    else:
        return frame, W, H


# convert openpose keypoints(25) format to coco keypoints(17) format
def convert17(op_kpts):
    '''
    (N, 25, 3) -> (N, 18, 3)
    '''
    coco_kpts = np.zeros(shape=(len(op_kpts), 17, 3))
    for item in range(len(op_kpts)):
        tmp = op_kpts[item][[0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11]]
        coco_kpts[item] = tmp

    return coco_kpts

# convert openpose keypoints(25) format to openpose coco keypoints(18) format
def convert18(op_kpts):
    '''
    (N, 25, 3) -> (N, 18, 3)
    '''
    coco_kpts = np.zeros(shape=(len(op_kpts), 18, 3))
    for item in range(len(op_kpts)):
        tmp = op_kpts[item][[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]
        coco_kpts[item] = tmp

    return coco_kpts


def draw_2Dimg(img, kpts, display=None, wait=1000, text=None):
    # kpts : (18, 3)  3-->(x, y, score)
    im = img.copy()
    joint_pairs = common.openpose_joint_pairs

    for kpt in kpts:
        for item in kpt:
            score = item[-1]
            if score > 0.2:
                x, y = int(item[0]), int(item[1])
                cv2.circle(im, (x, y), 1, (255, 0, 0), 5)

        for pair in joint_pairs:
            j, j_parent = pair
            score = min(kpt[j][-1], kpt[j_parent][-1])
            if score < 0.2:
                continue
            pt1 = (int(kpt[j][0]), int(kpt[j][1]))
            pt2 = (int(kpt[j_parent][0]), int(kpt[j_parent][1]))
            cv2.line(im, pt1, pt2, (255,255,0), 2)

    if text != None:
        #  print#  ('add text')
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,500)
        fontScale              = 1
        fontColor              = (255,255,255)
        lineType               = 2

        cv2.putText(im,text,
            bottomLeftCornerOfText,
            font,
            fontScale,
            fontColor,
            lineType)

    if display:
        cv2.imshow('im', im)
        cv2.waitKey(wait)
    return im


def videoInfo(VideoName):
    cap = cv2.VideoCapture(VideoName)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, length

