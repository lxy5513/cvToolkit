import sys
import os
import cv2
import numpy as np

joint_names = ['right ankle', 'right knee', 'right pelvis', 'left pelvis',                   'left knee', 'left ankle', 'right wrist',                   'right elbow', 'right shoulder', 'left shoulder', 'left elbow', 'left wrist',                   'upper neck', 'nose', 'head']
joint_pairs = [['head', 'upper neck', 'purple'],                   ['upper neck', 'right shoulder', 'yellow'],                   ['upper neck', 'left shoulder', 'yellow'],                   ['right shoulder', 'right elbow', 'blue'],                   ['right elbow', 'right wrist', 'green'],                   ['left shoulder', 'left elbow', 'blue'],                   ['left elbow', 'left wrist', 'green'],                   ['right shoulder', 'right pelvis', 'yellow'],                   ['left shoulder', 'left pelvis', 'yellow'],                   ['right pelvis', 'right knee', 'red'],                   ['right knee', 'right ankle', 'skyblue'],                   ['left pelvis', 'left knee', 'red'],                   ['left knee', 'left ankle', 'skyblue']]
color_list = ['purple', 'yellow', 'blue', 'green', 'red', 'skyblue', 'navyblue', 'azure', 'slate', 'chocolate', 'olive', 'orange', 'orchid']

flag_color_sticks = True


def show_poses_from_python_data(img, joints, joint_pairs=joint_pairs, joint_names=joint_names, flag_demo_poses = False, track_id = -1, flag_only_draw_sure=False):
    img = add_joints_to_image(img, joints)

    if track_id == -1: # do pose estimation visualization
        img = add_joint_connections_to_image(img, joints, joint_pairs, joint_names)
    else:  # do pose tracking visualization
        candidate_joint_pairs = joint_pairs.copy()
        color_name = color_list[track_id % 13]
        for i in range(len(candidate_joint_pairs)):   candidate_joint_pairs[i][2] = color_name
        img = add_joint_connections_to_image(img, joints, candidate_joint_pairs, joint_names, flag_only_draw_sure)

    if flag_demo_poses is True:
        cv2.imshow("pose image", img)
        cv2.waitKey(0.1)
    return img

def add_joints_to_image(img_demo, joints):
    for joint in joints:
        [i, j, sure] = joint
        #cv2.circle(img_demo, (i, j), radius=8, color=(255,255,255), thickness=2)
        cv2.circle(img_demo, (i, j), radius=2, color=(255,255,255), thickness=2)
    return img_demo


def add_joint_connections_to_image(img_demo, joints, joint_pairs, joint_names, flag_only_draw_sure = False):
    for joint_pair in joint_pairs:
        ind_1 = joint_names.index(joint_pair[0])
        ind_2 = joint_names.index(joint_pair[1])
        if flag_color_sticks is True:
            color = find_color_scalar(joint_pair[2])
        else:
            color = find_color_scalar('red')

        x1, y1, sure1 = joints[ind_1]
        x2, y2, sure2 = joints[ind_2]

        if x1 <= 5 and y1<= 5: continue
        if x2 <= 5 and y2<= 5: continue

        if flag_only_draw_sure is False:
            sure1 = sure2 = 1
        if sure1 > 0.8 or sure2 > 0.8:
            #cv2.line(img_demo, (x1, y1), (x2, y2), color, thickness=8)
            cv2.line(img_demo, (x1, y1), (x2, y2), color, thickness=4)
    return img_demo


def find_color_scalar(color_string):
    color_dict = {
        'purple': (255, 0, 255),
        'yellow': (0, 255, 255),
        'blue':   (255, 0, 0),
        'green':  (0, 255, 0),
        'red':    (0, 0, 255),
        'skyblue':(235,206,135),
        'navyblue': (128, 0, 0),
        'azure': (255, 255, 240),
        'slate': (255, 0, 127),
        'chocolate': (30, 105, 210),
        'olive': (112, 255, 202),
        'orange': (0, 140, 255),
        'orchid': (255, 102, 224)
    }
    color_scalar = color_dict[color_string]
    return color_scalar


def reshape_keypoints_into_joints(pose_keypoints_2d):
    # reshape vector of length 3N into an array of shape [N, 3]
    num_keypoints = int(len(pose_keypoints_2d) / 3)
    joints = np.array(pose_keypoints_2d).reshape(num_keypoints, 3)
    return joints

def videoInfo(VideoName):
    cap = cv2.VideoCapture(VideoName)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, length


class common():
    joint_pairs17 = [[0, 1], [1, 3], [0, 2], [2, 4],
                [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
                [5, 11], [6, 12], [11, 12],
                [11, 13], [12, 14], [13, 15], [14, 16]]

    joint_pairs18 = [[0, 1],[0, 14], [0, 15], [14, 16], [15, 17], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7]
            , [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13]]

# convert openpose keypoints(25) format to openpose coco keypoints(18) format
def convert18(op_kpts):
    '''
    (N, 25, 3) -> (N, 18, 3)
    '''
    if op_kpts.ndim == 2:
        op_kpts = np.expand_dims(op_kpts, 0)

    coco_kpts = np.zeros(shape=(len(op_kpts), 18, 3))
    for item in range(len(op_kpts)):
        tmp = op_kpts[item][[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]
        coco_kpts[item] = tmp
    return coco_kpts

def convert18_item(op_kpts):
    '''
    (25, 3) -> (18, 3)
    '''
    coco_kpts = op_kpts[[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]
    return coco_kpts

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
            if score < 0.1:
                continue 
            #  if int(kpt[j][0]) == 0:
                #  continue 
            #  if int(kpt[j][1]) == 0:
                #  continue 
            #  if int(kpt[j_parent][1]) == 0:
                #  continue 
            #  if int(kpt[j_parent][0]) == 0:
                #  continue 
            pt1 = (int(kpt[j][0]), int(kpt[j][1]))
            pt2 = (int(kpt[j_parent][0]), int(kpt[j_parent][1]))
            cv2.line(im, pt1, pt2, (255,255,0), 2)

    if text != None:
        #  print  ('add text')
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (30,30)
        fontScale              = 1.8
        fontColor              = (2,200,2)
        lineType               = 3

        cv2.putText(im,text,
            bottomLeftCornerOfText,
            font,
            fontScale,
            fontColor,
            lineType)

    if display:
        cv2.imshow('draw_img18', im)
        cv2.waitKey(wait)
    return im

