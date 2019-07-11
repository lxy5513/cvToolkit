import time
import collections

# import vision essentials
import cv2
import numpy as np
import ipdb; pdb=ipdb.set_trace
import sys, os, time

# import GCN utils
from graph import visualize_pose_matching
from graph  .visualize_pose_matching import *

# import my own utils
sys.path.append(os.path.abspath("./graph/"))
from utils_json import *
from visualizer import *
from utils_io_file import *
from utils_io_folder import *

from kps2d_detection.pose_utils import resize_img
from object_detection.yolo_v3.human_detector import load_model as load_yolo_model
from object_detection.yolo_v3.human_detector import inference

from utils_track import *
from utils_costum import draw_bbox

flag_nms = False #Default is False, unless you know what you are doing
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def initialize_parameters():
    global video_name, img_id
    global nms_method, nms_thresh, min_scores, min_box_size
    nms_method = 'nms'
    nms_thresh = 1.
    min_scores = 1e-10
    min_box_size = 0.

    global keyframe_interval, enlarge_scale, pose_matching_threshold
    keyframe_interval = 20 # choice examples: [2, 3, 5, 8, 10]
    enlarge_scale = 0.2
    pose_matching_threshold = 0.5

    global flag_flip
    flag_flip = True

    global total_time_POSE, total_time_DET, total_time_ALL, total_num_FRAMES, total_num_PERSONS
    total_time_POSE = 0
    total_time_DET = 0
    total_time_ALL = 0
    total_num_FRAMES = 0
    total_num_PERSONS = 0
    return

def get_human_bbox(image, model):
    bboxs, probs = inference(image, model)
    human_candidates = []
    for item in bboxs:
        x1,y1,x2,y2 = item
        w, h = x2-x1, y2-y1
        human_candidates.append([x1, y1, w, h])
    return human_candidates

FRAME_RECORD = 0
def light_track_camera(pose_estimator, video_capture):

    global total_time_POSE, total_time_DET, total_time_ALL, total_num_FRAMES, total_num_PERSONS
    ''' statistics: get total time for lighttrack processing'''
    st_time_total = time.time()

    # process the frames sequentially
    keypoints_list = []
    bbox_dets_list = []

    bbox_dets_list_q = collections.deque(maxlen=2)
    keypoints_list_q = collections.deque(maxlen=2)

    next_id = 0
    img_id = -1

    flag_mandatory_keyframe = False

    yolo_model = load_yolo_model()

    while video_capture.isOpened():
        img_id += 1
        total_num_FRAMES += 1

        ret, cur_img = video_capture.read()
        if cur_img is None: break
        cur_img, _, _ = resize_img(cur_img, 960)

        if img_id == 0:
            from cv_utils import video_out
            out = video_out(video_capture, cur_img, 'result.mp4')

        ''' KEYFRAME: (1) call the detector;
                      (2) perform HPE on the candidates;
                      (3) perform data association via Spatial Consistency and Pose Matching'''

        if is_keyframe(img_id, keyframe_interval) or flag_mandatory_keyframe:
            flag_mandatory_keyframe = False
            bbox_dets_list = []  # keyframe: start from empty
            keypoints_list = []  # keyframe: start from empty

            ## 1. FIND ALL HUMAN BBOX
            # perform detection at keyframes
            st_time_detection = time.time()
            human_candidates = get_human_bbox(cur_img, yolo_model)
            end_time_detection = time.time()
            total_time_DET += (end_time_detection - st_time_detection)

            num_dets = len(human_candidates)
            print("Keyframe: {} detections".format(num_dets))

            # if nothing detected at keyframe, regard next frame as keyframe because there is nothing to track
            if num_dets <= 0:
                # add empty result
                bbox_det_dict = {"img_id":img_id,
                                 "det_id":  0,
                                 "track_id": None,
                                 "bbox": [0, 0, 2, 2]}
                bbox_dets_list.append(bbox_det_dict)

                keypoints_dict = {"img_id":img_id,
                                  "det_id": 0,
                                  "track_id": None,
                                  "bbox": [0, 0, 2, 2],
                                  "keypoints": []}
                keypoints_list.append(keypoints_dict)

                bbox_dets_list_q.append(bbox_dets_list)
                keypoints_list_q.append(keypoints_list)

                flag_mandatory_keyframe = True
                cv2.imshow('frame', cur_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            ''' 2. statistics: get total number of detected persons '''
            total_num_PERSONS += num_dets

            if img_id > 0:   # First frame does not have previous frame
                bbox_list_prev_frame = bbox_dets_list_q.popleft()
                keypoints_list_prev_frame = keypoints_list_q.popleft()

            # For each candidate, perform pose estimation and data association based on Spatial Consistency (SC)
            for det_id in range(num_dets):
                # obtain bbox position and track id
                bbox_det = human_candidates[det_id]

                # enlarge bbox by 20% with same center position
                bbox_x1y1x2y2 = xywh_to_x1y1x2y2(bbox_det)
                bbox_in_xywh = enlarge_bbox(bbox_x1y1x2y2, enlarge_scale)
                bbox_det = x1y1x2y2_to_xywh(bbox_in_xywh)

                # Keyframe: use provided bbox
                if bbox_invalid(bbox_det): #无效的bbox
                    track_id = None # this id means null
                    keypoints = []
                    bbox_det = [0, 0, 2 ,2]
                    # update current frame bbox
                    bbox_det_dict = {"img_id":img_id,
                                     "det_id":det_id,
                                     "track_id": track_id,
                                     "bbox":bbox_det}
                    bbox_dets_list.append(bbox_det_dict)
                    # update current frame keypoints
                    keypoints_dict = {"img_id":img_id,
                                      "det_id":det_id,
                                      "bbox":bbox_det,
                                      "track_id": track_id,
                                      "keypoints":keypoints}
                    keypoints_list.append(keypoints_dict)
                    continue

                # update current frame bbox
                bbox_det_dict = {"img_id":img_id,
                                 "det_id":det_id,
                                 "bbox":bbox_det}

                # obtain keypoints for each bbox position in the keyframe
                st_time_pose = time.time()
                keypoints = inference_keypoints_custom(pose_estimator, bbox_det_dict, cur_img, get_keypoints_from_bbox)[0]["keypoints"]
                end_time_pose = time.time()
                total_time_POSE += (end_time_pose - st_time_pose)

                if img_id == 0:   # First frame, all ids are assigned automatically
                    track_id = next_id
                    next_id += 1
                else:
                    #### 根据IOU得到本帧的一个bbox_det的track_id
                    track_id, match_index = get_track_id_SpatialConsistency(bbox_det, bbox_list_prev_frame)
                    if track_id != -1:  # if candidate from prev frame matched, prevent it from matching another
                        del bbox_list_prev_frame[match_index]
                        del keypoints_list_prev_frame[match_index]

                # update current frame bbox
                bbox_det_dict = {"img_id":img_id,
                                 "det_id":det_id,
                                 "track_id":track_id,
                                 "bbox":bbox_det}
                bbox_dets_list.append(bbox_det_dict)

                # update current frame keypoints
                keypoints_dict = {"img_id":img_id,
                                  "det_id":det_id,
                                  "bbox":bbox_det,
                                  "track_id":track_id,
                                  "keypoints":keypoints}
                keypoints_list.append(keypoints_dict)

            ######## 处理所有bbox IOU 未匹配成功的
            # For candidate that is not assopciated yet, perform data association based on Pose Similarity (SGCN)
            for det_id in range(num_dets):
                bbox_det_dict = bbox_dets_list[det_id]
                keypoints_dict = keypoints_list[det_id]
                assert(det_id == bbox_det_dict["det_id"])
                assert(det_id == keypoints_dict["det_id"])

                if bbox_det_dict["track_id"] == -1:    # this id means matching not found yet
                    track_id, match_index = get_track_id_SGCN(bbox_det_dict["bbox"], bbox_list_prev_frame,
                                                                 keypoints_dict["keypoints"], keypoints_list_prev_frame, pose_matching_threshold=pose_matching_threshold)

                    if track_id != -1:  # if candidate from prev frame matched, prevent it from matching another
                        del bbox_list_prev_frame[match_index]
                        del keypoints_list_prev_frame[match_index]
                        bbox_det_dict["track_id"] = track_id
                        keypoints_dict["track_id"] = track_id

                    # if still can not find a match from previous frame, then assign a new id
                    if track_id == -1 and not bbox_invalid(bbox_det_dict["bbox"]):
                        bbox_det_dict["track_id"] = next_id
                        keypoints_dict["track_id"] = next_id
                        next_id += 1

            # update frame
            bbox_dets_list_q.append(bbox_dets_list)
            keypoints_list_q.append(keypoints_list)


        ## 非关键帧
        else:
            ''' NOT KEYFRAME: (1) perform Single Pose Tracking (SPT) and Single Object Tracking (SOT)
                                  via Human Pose Estimation (HPE) for each candidate; '''
            bbox_dets_list_next = []
            keypoints_list_next = []

            num_dets = len(keypoints_list)
            total_num_PERSONS += num_dets
            #  print("Non-Keyframe: tracking {} candidates".format(num_dets))

            if num_dets == 0:
                flag_mandatory_keyframe = True

            for det_id in range(num_dets):
                keypoints = keypoints_list[det_id]["keypoints"]

                # for non-keyframes, the tracked target preserves its track_id
                track_id = keypoints_list[det_id]["track_id"]

                # next frame bbox 可以考虑替换为graph flow
                bbox_det_next = get_bbox_from_keypoints(keypoints, enlarge_scale)
                if bbox_det_next[2] == 0 or bbox_det_next[3] == 0:
                    bbox_det_next = [0, 0, 2, 2]
                    total_num_PERSONS -= 1
                assert(bbox_det_next[2] != 0 and bbox_det_next[3] != 0) # width and height must not be zero
                bbox_det_dict_next = {"img_id":img_id,
                                     "det_id":det_id,
                                     "track_id":track_id,
                                     "bbox":bbox_det_next}

                # next frame keypoints
                st_time_pose = time.time()
                keypoints_next = inference_keypoints_custom(pose_estimator, bbox_det_dict_next, cur_img, get_keypoints_from_bbox)[0]["keypoints"]
                end_time_pose = time.time()
                total_time_POSE += (end_time_pose - st_time_pose)
                #print("time for pose estimation: ", (end_time_pose - st_time_pose))

                # check whether the target is lost 有没有匹配
                target_lost = is_target_lost(keypoints_next)

                if target_lost is False:
                    bbox_dets_list_next.append(bbox_det_dict_next)
                    keypoints_dict_next = {"img_id":img_id,
                                           "det_id":det_id,
                                           "bbox":bbox_det_next,
                                           #  "bbox":bbox_det,
                                           "track_id":track_id,
                                           "keypoints":keypoints_next}
                    keypoints_list_next.append(keypoints_dict_next)

                # 目标消失
                else:
                    # remove this bbox, do not register its keypoints
                    bbox_det_dict_next = {"img_id":img_id,
                                          "det_id":  det_id,
                                          "track_id": -1,
                                          "bbox": [0, 0, 2, 2]}
                    bbox_dets_list_next.append(bbox_det_dict_next)

                    keypoints_null = 45*[0]
                    keypoints_dict_next = {"img_id":img_id,
                                           "det_id":det_id,
                                           "bbox": [0, 0, 2, 2],
                                           "track_id":track_id,
                                           "keypoints": []}
                    keypoints_list_next.append(keypoints_dict_next)
                    print("Target lost. Process this frame again as keyframe. \n\n\n")
                    flag_mandatory_keyframe = True

                    ## Re-process this frame by treating it as a keyframe
                    total_num_PERSONS -= 1
                    if img_id not in [0]:
                        img_id -= 1

                    # Re-process if anyone of the targets is lost
                    break

            # update frame
            if flag_mandatory_keyframe is False:
                bbox_dets_list = bbox_dets_list_next
                keypoints_list = keypoints_list_next
                bbox_dets_list_q.append(bbox_dets_list)
                keypoints_list_q.append(keypoints_list)
                bbox_dets_list_q.popleft()
                keypoints_list_q.popleft()

        # visulize this frame
        cur_candidates = list(keypoints_list_q)[-1]  # peek right-most item, latest
        change_color = False
        if flag_mandatory_keyframe or img_id%keyframe_interval==0:
            # 关键帧颜色变黑
            change_color = True
        vis_img = visualize_img(cur_img, cur_candidates, img_id, change_color=change_color)

        # provide a way to exit
        cv2.imshow('frame', vis_img)
        out.write(vis_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        #  if change_color:
            #  print("KEY FRAME .............")
            #  time.sleep(1)

    # release resources
    video_capture.release()
    cv2.destroyAllWindows()

    ''' statistics: get total time for lighttrack processing'''
    end_time_total = time.time()
    total_time_ALL += (end_time_total - st_time_total)


def visualize_img(img, candidates, img_id, change_color=None):
    for candidate in candidates:
        bbox = np.array(candidate["bbox"]).astype(int)

        # optional: show the bounding boxes
        track_id = candidate["track_id"]
        if track_id == 0:
            text_id = '0'
        else:
            text_id = track_id

        img = draw_bbox(img, bbox, text=text_id, change_color=change_color)

        pose_keypoints_2d = candidate["keypoints"]
        joints = reshape_keypoints_into_joints(pose_keypoints_2d)

        track_id = candidate["track_id"]
        img = show_poses_from_python_data(img, joints, joint_pairs, joint_names, track_id = track_id, flag_only_draw_sure = True)
    else:
        img = show_poses_from_python_data(img, joints, joint_pairs, joint_names)
    return img


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', '-v', type=str, dest='video_path', default="/path/to/cvToolBox/data/football.mp4")
    parser.add_argument('--pose_detector', '-p', type=int, default=0)
    args = parser.parse_args()
    args.bbox_thresh = 0.4
    initialize_parameters()

    pose_detector = args.pose_detector  
    if pose_detector == 0: #simplebaseline  
        print('Use simple baseline as pose estimator')
        from kps2d_detection.simple_baseline.pose_estimation import get_pose_model, get_keypoints_from_bbox
    else:
        print('Use hrnet as pose estimator')
        from kps2d_detection.hr_net.pose_estimation import get_pose_model, get_keypoints_from_bbox
        
    pose_estimator = get_pose_model()
    video_name = args.video_path
    video_capture = cv2.VideoCapture(video_name)
    if not video_capture.isOpened():
        video_capture.open()

    if video_capture.isOpened():
        light_track_camera(pose_estimator, video_capture)
        print("Finished Camera Demo")

        ''' Display statistics '''
        print("total_num_FRAMES: {:d}".format(total_num_FRAMES))
        print("total_num_PERSONS: {:d}\n".format(total_num_PERSONS))
        print("Average FPS: {:.2f}fps".format(total_num_FRAMES / total_time_ALL))
        print("AVERAGE PFS FOR EACH POSE IS {:.2f}fps".format(total_num_PERSONS/total_time_POSE))
    else:
        print("Camera not found.")
