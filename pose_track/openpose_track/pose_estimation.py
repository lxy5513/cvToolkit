import sys
import cv2
import os
from sys import platform
import argparse
from tqdm import tqdm
import ipdb;pdb=ipdb.set_trace
import numpy as np
try:
    sys.path.append('/usr/local/python')
    from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

params = dict()
params["model_folder"] = "/openpose/models/"
params['num_gpu'] = 1
params['num_gpu_start'] = 1

def convert18(op_kpts):
    '''
    (N, 25, 3) -> (N, 18, 3)
    '''
    coco_kpts = np.zeros(shape=(len(op_kpts), 18, 3))
    for item in range(len(op_kpts)):
        tmp = op_kpts[item][[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]
        coco_kpts[item] = tmp
    return coco_kpts

def get_model(tracking=0):
    # Tracking
    if tracking:
        params["tracking"] = 5
        params["number_people_max"] = 1
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    print("get model ---> done")
    return opWrapper

def get_keypoints_from_openpose(model, img):
    datum = op.Datum()
    datum.cvInputData = img
    model.emplaceAndPop([datum])
    kpts = datum.poseKeypoints
    return kpts

def handle_video(video_name):
    import cv2
    cap = cv2.VideoCapture(video_name)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    model = load_model()
    for i in tqdm(range(length)):
        _, frame = cap.read()
        kps = interface(model, frame)
        draw_2Dimg(frame, convert18(kps), 1, 3)

if __name__ == '__main__':
    import cv2
    im = cv2.imread('../test.png')
    model = get_model(1)
    kps = get_keypoints(model, im)
    kps = convert18(kps)
    #  draw_2Dimg(im, kps, 1)

    #  import os
    #  video_name = os.path.join(os.environ.get("CVTOOLBOX"), 'data/test.mp4')
    #  handle_video(video_name)
