import json
import cv2
import numpy as np
import os
import ipdb;pdb=ipdb.set_trace

class common():

    joint_pairs17 = [[0, 1], [1, 3], [0, 2], [2, 4],
                [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
                [5, 11], [6, 12], [11, 12],
                [11, 13], [12, 14], [13, 15], [14, 16]]

    joint_pairs18 = [[0, 1],[0, 14], [0, 15], [14, 16], [15, 17], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7]
            , [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13]]


def resize_img(frame, max_length=640):
    H, W = frame.shape[:2]
    if max(W, H) > max_length: #shrink
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LINEAR


    if W>H:
        W_resize = max_length
        H_resize = int(H * max_length / W)
    else:
        H_resize = max_length
        W_resize = int(W * max_length / H)
    frame = cv2.resize(frame, (W_resize, H_resize), interpolation=interpolation)
    return frame, W_resize, H_resize


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


def draw_img18(im, kpts, display=None, wait=1000, text=None):
    # kpts : (N, 18, 3)  3-->(x, y, score)
    if kpts.ndim == 2:
        kpts = np.expand_dims(kpts, 0)

    joint_pairs = common.joint_pairs18
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
        #  print  ('add text')
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (30,30)
        fontScale              = 0.8
        fontColor              = (2,2,2)
        lineType               = 1

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



def draw_img17(im, kpts, display=None, wait=1000, text=None):
    # kpts : (N, 17, 3)  3-->(x, y, score)
    if kpts.ndim == 2:
        kpts = np.expand_dims(kpts, 0)

    joint_pairs = common.joint_pairs17
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
