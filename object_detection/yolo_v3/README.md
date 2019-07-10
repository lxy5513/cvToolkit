## use for image human bounding box obtain


## 检测人体  返回人体box (x1,y1,x2,y2) 和 概率
```
from human_detector import load_model
from human_detector import inference
import cv2
im = cv2.imread('/home/xyliu/Pictures/Figure_1.png')
model = load_model()
bbox, probs = inference(im, model)
```
