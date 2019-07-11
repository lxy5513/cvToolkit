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
    - [x]lighttrack

- ActionRecgnition 
    - ST-GCN  
    - 2S-AGCN



## Install  
> Install pytorch >= v1.0.0  
pip install -r requirment.txt  

## Make  
> cd kps2d_detection/hr_net/lib/  
make  


## Example of Human Bbox Detection  
    ```   
    from object_detection.yolo_v3.human_detector import load_model
    from object_detection.yolo_v3.human_detector import inference
    import cv2
    im = cv2.imread('/home/xyliu/Pictures/Figure_1.png')
    model = load_model()
    # return bbox and its probability
    inference(im, model)
    ```

