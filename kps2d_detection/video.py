import os
import cv2
from tqdm import tqdm
from pose_utils import draw_img17, resize_img, draw_img18, convert18
import ipdb;pdb=ipdb.set_trace


def simple_baseline_video(video_name, display=None):
    from simple_baseline.pose_estimation import get_keypoints, get_two_model
    m1, m2 = get_two_model()
    cap = cv2.VideoCapture(video_name)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # warm up
    _, frame = cap.read()
    for _ in range(15):
        frame, W, H = resize_img(frame, 960)
        kps = get_keypoints(m1, m2, frame)

    for i in tqdm(range(length-1)):
        _, frame = cap.read()
        frame, W, H = resize_img(frame, 960)
        kps = get_keypoints(m1, m2, frame)
        if display:
            draw_img17(frame, kps, 1, 3)

def hr_net_video(video_name, display=None):
    from hr_net.pose_estimation import get_two_model, get_keypoints
    m1, m2 = get_two_model()
    cap = cv2.VideoCapture(video_name)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # warm up
    _, frame = cap.read()
    for _ in range(15):
        frame, W, H = resize_img(frame)
        kps = get_keypoints(m1, m2, frame)

    for i in tqdm(range(length-1)):
        _, frame = cap.read()
        frame, W, H = resize_img(frame)
        kps = get_keypoints(m1, m2, frame)
        if display:
            draw_img17(frame, kps, 1, 3)


def openpose_video(video_name, display=None):
    from open_pose.pose_estimation import get_model, get_keypoints
    op_model = get_model(tracking=1)
    cap = cv2.VideoCapture(video_name)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # warm up
    _, frame = cap.read()
    for _ in range(15):
        frame, W, H = resize_img(frame)
        kps = get_keypoints(op_model, frame)

    for i in tqdm(range(length-1)):
        _, frame = cap.read()
        frame, W, H = resize_img(frame)
        kps = get_keypoints(op_model, frame)
        kps = convert18(kps)
        if display:
            draw_img18(frame, kps, 1, 3)


if __name__ == "__main__":
    video_name = os.path.join(os.environ.get("CVTOOLBOX"), 'data/football.mp4')

    simple_baseline_video(video_name, 1) #31 FPS
    #  hr_net_video(video_name, 1) #19 FPS
    #  openpose_video(video_name) #52 FPS (2 GPU)

