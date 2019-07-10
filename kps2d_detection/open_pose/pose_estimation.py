import sys
import cv2
import os
from sys import platform
import argparse
from tqdm import tqdm
import ipdb;pdb=ipdb.set_trace
try:
    from .utils import draw_2Dimg, convert18
except:
    from utils import draw_2Dimg, convert18

# Import Openpose (Windows/Ubuntu/OSX) /home/xyliu/2D_pose/openpose
dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

print(dir_path)
params = dict()
params["model_folder"] = dir_path + "/models/"

def get_model(tracking=0):
    if tracking:
        params["tracking"] = 5
        params["number_people_max"] = 1
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    return opWrapper

def get_keypoints(model, img):
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
    im = cv2.imread('../../data/test.png')
    model = load_model()
    kps = interface(model, im)
    kps = convert18(kps)
    draw_2Dimg(im, kps, 1)

    import os
    video_name = os.path.join(os.environ.get("CVTOOLBOX"), 'data/test.mp4')
    handle_video(video_name)
