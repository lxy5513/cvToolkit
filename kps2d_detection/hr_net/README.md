##  pose estimation by deep-high-resolution network
```  
from pose_estimation import get_two_model, get_keypoints, draw_img
m1, m2 = get_two_model()
import cv2
im = cv2.imread('/home/xyliu/Pictures/Figure_1.png')
result = get_keypoints(m1, m2, im)
draw_img(im, result, 1)
```  

## install  
pip install -r requirements.txt  
cd lib  
make   

