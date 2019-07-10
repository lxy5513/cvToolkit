import os
import cv2
from tqdm import tqdm
from pose_utils import draw_img17, resize_img, draw_img18, convert18
import ipdb;pdb=ipdb.set_trace
from argparse import ArgumentParser
import numpy as np
MAXTORSO = 0
def action_recgnition(kps, thred=0.1):
    global MAXTORSO
    average_score = kps[:,2].sum() / 17
    assert average_score > thred, 'average score is too low'

    hip = np.zeros((2,3))
    hipLeft, hipRight = kps[8], kps[11]
    hip[0], hip[1] = kps[8], kps[11]
    mask = hip[:, 2] > thred
    assert mask.sum()>0, 'hip score too low'
    hipHight = (hip[:, 1] * mask).sum() / mask.sum()
    throat = kps[1]
    assert throat[2]>thred, 'thaort score is too low'
    hipCenter =  np.sum(hip, axis=0)/2
    torsoLength = np.linalg.norm(throat[:2]-hipCenter[:2])
    if MAXTORSO<torsoLength:
        MAXTORSO=torsoLength
    print('torso length is ', MAXTORSO)


    head = np.zeros((5, 3))
    head[0] = kps[0]  # nose
    head[1:] = kps[14:18]  # leftEye rightEye leftEar rightEar
    mask = head[:,2] > thred
    headHeight = (head[:, 1] * mask).sum() / mask.sum()

    foot = np.zeros((2, 3))
    foot[0], foot[1] = kps[10], kps[13]
    mask = foot[:, 2] > thred
    assert mask.sum()>0, 'foot score is too low'
    footHeight = (foot[:, 1] * mask).sum() / mask.sum()

    handLeft, handRight = kps[4], kps[7]
    #  assert handLeft[2]>thred and handRight[2]>thred, 'two hand score are too low'


    knee = np.zeros((2,3))
    kneeRight, kneeLeft = kps[9], kps[12]
    knee[0], knee[1] = kneeLeft, kneeRight
    assert kneeRight[2]>thred and kneeLeft[2]>thred, 'two knee score are too low'
    mask = knee[:, 2] > thred
    kneeHeight = (knee[:, 1] * mask).sum() / mask.sum()


    # fall
    torsoLength=MAXTORSO
    print('FALL:   {}, {}'.format((footHeight-headHeight), torsoLength)  )
    print('squat', abs(hipHight-kneeHeight), torsoLength/5*2, footHeight-hipHight, torsoLength)
    if footHeight-headHeight < torsoLength:
        print('fall')
        return 'fall'
    elif headHeight-handLeft[1] > 0 and headHeight-handRight[1] > 0:
        print('hand up and help!')
        return 'hand up and help'
    elif abs(hipHight-kneeHeight) < torsoLength/5*2 and footHeight-hipHight>torsoLength/2:
        print('squat')
        return 'squat'
    else:
        print('Other')
        return 'other'


def openpose_video(video_name, display=None):
    from open_pose.pose_estimation import get_model, get_keypoints
    op_model = get_model(tracking=1)
    cap = cv2.VideoCapture(video_name)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame = cap.read()
    frame, W, H = resize_img(frame)
    H,W = frame.shape[:2]
    output_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # ⚠️读取时是（高、宽、3）， 写入是设置是（宽、高）
    out = cv2.VideoWriter('result.mp4',fourcc, output_fps, (W,H))

    if length<10:
        length = 10000

    for i in tqdm(range(length-1)):
        _, frame = cap.read()
        frame, W, H = resize_img(frame)
        kps = get_keypoints(op_model, frame)
        kps = convert18(kps)

        #  if i>=40 and i%5==0:
            #  pdb()
        try:
            action_text=action_recgnition(kps[0])
            print('Action Text is ', action_text)
            if display:
                img = draw_img18(frame, kps, 1, 3, text=action_text)
                out.write(img)

        except Exception as e:
            print(e)

if __name__ == "__main__":
    video_name = os.path.join(os.environ.get("CVTOOLBOX"), 'data/test.mp4')


    parser = ArgumentParser()
    parser.add_argument("-i", "--video_input", help="input video file name", default="/home/xyliu/Videos/sports/dance.mp4")
    args = parser.parse_args()
    video_name = args.video_input

    openpose_video(video_name, 1) #52 FPS (2 GPU)

