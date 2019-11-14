import time
import collections
import requests
import asyncio
from tqdm import tqdm
import cv2
import numpy as np
import ipdb; pdb=ipdb.set_trace
import sys, os, time
from myutils.utils_track import get_track_id_SGCN_plus, is_keyframe, is_target_lost, get_track_id_SpatialConsistency, get_bbox_from_keypoints
from myutils.utils_pose import convert17_18
from myutils.utils_vis import reshape_keypoints_into_joints, show_poses_from_python_data, videoInfo
from myutils.utils_img import xywh_to_x1y1x2y2, x1y1x2y2_to_xywh, draw_bbox, enlarge_bbox, bbox_invalid, resize_img
from pose_estimation import get_model, get_keypoints_from_openpose
from myutils.utils_vis import convert18, convert18_item
joint_pairs = [[0, 1],[0, 14], [0, 15], [14, 16], [15, 17], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13]]
now = lambda:time.time()
FALL_THRED = 0.95
valid_points_thred = 10 
valid_area_thred = 0.02 

def video_out(cap, img, save_name='video_out.mp4'):
    # out.write(img)
    H,W = img.shape[:2]
    output_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_name,fourcc, output_fps, (W,H))
    return out

def get_info_from_openpose(kpts, thred=0.4):
    """ 
    1. get x1y1x2y2 coordinates 
    2. get valid points number 
    3. get area of keypoints 
    """
    x_list = []
    y_list = []
    valid_num = 0 # valis keypoints number 
    for x, y, score in kpts:
        if score > thred:
            x_list.append(x)
            y_list.append(y)
            valid_num += 1
    min_x = min(x_list)
    min_y = min(y_list)
    max_x = max(x_list)
    max_y = max(y_list)
    area = (max_x-min_x) * (max_y-min_y)
    return [min_x, min_y, max_x, max_y], valid_num, int(area)

class PoseTrack():
    def __init__(self, enlarge_scale=0.2):
        self.enlarge_scale = enlarge_scale
        self.next_id = 0
        self.img_id = 0
        self.image = None
        self.dets_list_q = collections.deque(maxlen=5)
        self.time_rec = collections.deque(maxlen=20)
        self.kps_records = {}
        self.queue_length = 60 # the histoty coco keypoins max length
        self.op_model = get_model()


    def draw_result(self):
        candidates = self.dets_list_q[-1]
        image = self.image.copy()
        try:
            for candidate in candidates:
                track_id = candidate["track_id"]
                bbox = np.array(candidate["bbox"]).astype(int)
                pose_keypoints_2d = candidate["openpose_kps"]
                joints = reshape_keypoints_into_joints(pose_keypoints_2d)
                coco_kps = convert18_item(joints)
                image = draw_bbox(image, bbox, str(track_id))
                image = draw_img18(image, coco_kps)
        except Exception as e:
            pass

        image = resize_img(image, 480)[0]
        cv2.imshow("track result", image)
        cv2.waitKey(5)

    # MAIN FUNCTION
    def pose_track(self, image, img_id=None, dets_list_q=None):
        self.image = image

        ##### confirm next id ##### 
        if img_id != 0:
            self.img_id = img_id
            for i in range(len(self.dets_list_q)):
                index = -(i+1)
                prev_candidates = list(self.dets_list_q)[index]
                next_ids = [prev_candidates[item]['track_id'] for item in range(len(prev_candidates)) if prev_candidates[item]['track_id']!=None ]
                if next_ids != []:
                    self.next_id = max(max(next_ids)+1, self.next_id)

        self.posetrack()
        self.draw_result()
        self.img_id += 1
        return self.img_id

    def posetrack(self):
        t0 = now()
        human_candidates = self.get_human_bbox_and_keypoints()
        self.time_rec.append(now()-t0) 
        num_dets = len(human_candidates)
        if num_dets <= 0:
            dets_list = []
            det_dict = {"img_id":self.img_id,
                                "det_id":  0,
                                "track_id": None,
                                "openpose_kps": [],
                                "bbox": [0, 0, 2, 2],
                                "keypoints": []}
            dets_list.append(det_dict)
            self.dets_list_q.append(dets_list)
            return

        if self.img_id == 0:
            self.first_frame(human_candidates)
            return

        ##### traverse all prev frame dicts #####
        tracked_dets_list = []
        tracked_dets_ids = []
        untracked_dets_ids = list(range(len(human_candidates)))
        for i in range(len(self.dets_list_q)):
            index = -(i+0)
            dets_list_prev_frame = self.dets_list_q[index]
            if len(untracked_dets_ids) > 0:
                self.traverse_each_prev_frame(human_candidates, dets_list_prev_frame, tracked_dets_list,tracked_dets_ids,  untracked_dets_ids)
            untracked_dets_ids = list(set(untracked_dets_ids)-set(tracked_dets_ids))

        ##handle all unmatched item
        det_dict = {"img_id":self.img_id,
                            "det_id": 0,
                            "track_id": -1,
                            "bbox":[0,0,2,2],
                            "openpose_kps": [],
                            "keypoints": []}

        for det_id in untracked_dets_ids:
            bbox_det = human_candidates[det_id][0]
            bbox_x1y1x2y2 = xywh_to_x1y1x2y2(bbox_det)
            bbox_in_xywh = enlarge_bbox(bbox_x1y1x2y2, self.enlarge_scale)
            bbox_det = x1y1x2y2_to_xywh(bbox_in_xywh)
            openpose_kps = human_candidates[det_id][1]
            keypoints = convert25_15(openpose_kps)
            det_dict["track_id"] = self.next_id
            det_dict['det_id'] = det_id
            det_dict['openpose_kps'] = openpose_kps 
            det_dict['bbox'] = bbox_det
            self.next_id += 1
            tracked_dets_list.append(det_dict)
            
        self.dets_list_q.append(tracked_dets_list)

    def traverse_each_prev_frame(self, human_candidates, dets_list_prev_frame, tracked_dets_list, tracked_dets_ids, untracked_dets_ids):
        # first travese all bbox candidates
        for det_id in untracked_dets_ids:
            bbox_det = human_candidates[det_id][0]
            bbox_x1y1x2y2 = xywh_to_x1y1x2y2(bbox_det)
            bbox_in_xywh = enlarge_bbox(bbox_x1y1x2y2, self.enlarge_scale)
            bbox_det = x1y1x2y2_to_xywh(bbox_in_xywh)
            openpose_kps = human_candidates[det_id][1]
            keypoints = convert25_15(openpose_kps)
            det_dict = {"img_id":self.img_id,
                                "det_id":det_id,
                                "bbox":bbox_det,
                                "track_id": -1,
                                "openpose_kps": openpose_kps,
                                "keypoints":keypoints}

            track_id, match_index = get_track_id_SpatialConsistency(bbox_det, dets_list_prev_frame)
            if track_id != -1:  # if candidate from prev frame matched, prevent it from matching another
                del dets_list_prev_frame[match_index]
                det_dict['track_id'] = track_id
                tracked_dets_list.append(det_dict)
                tracked_dets_ids.append(det_id)
                continue 

        untracked_dets_ids = list(set(untracked_dets_ids)-set(tracked_dets_ids))
        # second travese all pose candidates
        for det_id in untracked_dets_ids:
            bbox_det = human_candidates[det_id][0]
            bbox_x1y1x2y2 = xywh_to_x1y1x2y2(bbox_det)
            bbox_in_xywh = enlarge_bbox(bbox_x1y1x2y2, self.enlarge_scale)
            bbox_det = x1y1x2y2_to_xywh(bbox_in_xywh)
            openpose_kps = human_candidates[det_id][1]
            keypoints = convert25_15(openpose_kps)
            det_dict = {"img_id":self.img_id,
                                "det_id":det_id,
                                "bbox":bbox_det,
                                "track_id": -1,
                                "openpose_kps": openpose_kps,
                                "keypoints":keypoints}
            track_id, match_index, score = get_track_id_SGCN_plus(det_dict, dets_list_prev_frame, pose_matching_threshold=0.4)
            if track_id != -1:  # if candidate from prev frame matched, prevent it from matching another
                print("match socre is" , score)
                del dets_list_prev_frame[match_index]
                det_dict["track_id"] = track_id
                tracked_dets_list.append(det_dict)
                tracked_dets_ids.append(det_id)
                continue
            untracked_dets_ids = list(set(untracked_dets_ids)-set(tracked_dets_ids))
            

    def first_frame(self, candidates):
        dets_list = []
        for i in range(len(candidates)):
            candidate = candidates[i]
            bbox = candidate[0]
            openpose_kps = candidate[1] 
            keypoints = convert25_15(openpose_kps)
            det_dict = {"img_id":self.img_id,
                                "det_id":  i,
                                "track_id": self.next_id,
                                "bbox": bbox,
                                "openpose_kps": openpose_kps,
                                "keypoints": keypoints}
            dets_list.append(det_dict)
            self.next_id += 1
        self.dets_list_q.append(dets_list)        


    def get_human_bbox_and_keypoints(self):
        kps = get_keypoints_from_openpose(self.op_model, self.image)
        human_candidates = []
        try:
            for kpt_item in kps:
                kpt_score = self.get_total_score_from_kpt(kpt_item)
                if kpt_score < 5:
                    continue
                kpt_item = list(kpt_item.reshape(-1))
                bbox = get_bbox_from_keypoints(kpt_item)
                human_candidate = [bbox, kpt_item]
                human_candidates.append(human_candidate)
        except:
            pass

        return human_candidates

    def get_total_score_from_kpt(self, kpt_item):
        scores = np.sum(kpt_item[...,[2]])
        return scores


def convert25_15(keypoints):
    # covert to poseTrack keypoints for SGCN(pose matching) 
    # [25 * 3] --> [15 * 3] [14 13 12 9 10 11 7 6 5 2 3 4 0 0 0 ]
    kpts = [] 
    list_pose_track = [14, 13, 12, 9, 10, 11, 7, 6, 5, 2, 3, 4, 0, 0, 0]
    for i in list_pose_track:
        kpts = kpts + [keypoints[i*3], keypoints[i*3+1], keypoints[i*3+2]] 
    return kpts



def draw_img18(im, kpt):
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
        pt1 = (int(kpt[j][0]), int(kpt[j][1]))
        pt2 = (int(kpt[j_parent][0]), int(kpt[j_parent][1]))
        cv2.line(im, pt1, pt2, (255,255,0), 2)
    return im

def draw_kpts(im, candidates):
    for candidate in candidates:
        bbox = candidate['bbox']
        track_id = candidate['track_id']
        pose_keypoints_2d = candidate["openpose_kps"]
        kpt = reshape_keypoints_into_joints(pose_keypoints_2d)
        kpt = convert18_item(kpt)
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
            pt1 = (int(kpt[j][0]), int(kpt[j][1]))
            pt2 = (int(kpt[j_parent][0]), int(kpt[j_parent][1]))
            cv2.line(im, pt1, pt2, (255,255,0), 2)
    return im

def main(path):
    now = time.time
    poseTrack=PoseTrack()
    cap, length = videoInfo(path)
    _, frame = cap.read()
    frame, _, _ = resize_img(frame, 960)
    out = video_out(cap, frame)
    time_rec = collections.deque(maxlen=20)
    img_id = 0
    for i in tqdm(range(length)):
        t0 = now()
        _, frame = cap.read()
        frame, _, _ = resize_img(frame, 960)
        img_id = poseTrack.pose_track(frame, img_id)
        out.write(frame)

if __name__ == '__main__':
    try:
        path = sys.argv[2]
    except Exception as e:
        path = "football.mp4"

    print("video path: ", path)
    main(path)

