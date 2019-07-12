# Computer Vision ToolBox of Pytorch

- ObjectDection
    Human Bbox Detection
    - [x] YoloV3
    - RFCN
    - CornerNet

- 2D pose estimation (kps2d__detection)
    - [x] Simple BaseLine
    - [x] High Resolution Network (hr_net)
    - [x] Open Pose

- 3D pose estimation (kps3d_detection)
    - VideoPose3D

- Object Tracking
    - [x] lighttrack
    - pose flow

- ActionRecgnition
    - ST-GCN
    - 2S-AGCN



## Install
> Install pytorch >= v1.0.0
pip install -r requirment.txt

## Make
> cd kps2d_detection/hr_net/lib/  
make   
> cd pose_track/lighttrack/graph/torchlight    
python setup.py install


## add Environment Variable 
vim ~/.bashrc
export PATH=/path/to/cvToolBox:$PATH
export PYTHONPATH=/path/to/cvToolBox:$PYTHONPATH

## Example of Human Bbox Detection
    ```
    from object_detection.yolo_v3.human_detector import load_model
    from object_detection.yolo_v3.human_detector import inference
    import cv2
    im = cv2.imread('data/test.png')
    model = load_model()
    # return bbox and its probability
    inference(im, model)
    ```

## Example of pose track

  > cd  pose_track/lighttrack/


   - python demo.py -p 0 (pose estimator is simple-baseline,  pose speed 90 person/s)
   <p align="center"><img src="pose_track/lighttrack/data/hrnet_result.gif" width="70%" alt="" /></p>


   - python demo.py -p 1 (pose estimator is hrnet,  pose speed 25 person/s)
   <p align="center"><img src="pose_track/lighttrack/data/hrnet_result.gif" width="70%" alt="" /></p>

