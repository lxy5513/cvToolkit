# Computer Vision ToolBox of Pytorch 

- ObjectDection  
    Human Bbox Detection  
    - [x] YoloV3 
    - RFCN  
    - FasterRCNN  
    - CornerNet

- 2D pose estimation  
    - [x] Simple Base Line 
    - [x] High Resolution Network  
    - [x] Ahlpha Pose   

- 3D pose estimation 
    - VideoPose3D  

- Object Tracking  
    - Human Tracking  
    - Flow Net

- ActionRecgnition 
    - ST-GCN  
    - AS-GCN  



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


## Install  
Install pytorch >= v1.0.0  
