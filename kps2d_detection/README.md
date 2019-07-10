# 2D keypoints detection  

## Hrnet 
High Resolution network    
```  
from hr_net.pose_estimation import get_two_model, get_keypoints, draw_img
m1, m2 = get_two_model()
import cv2
im = cv2.imread('../data/test.png')
result = get_keypoints(m1, m2, im)
draw_img(im, result, 1)

```

## 
