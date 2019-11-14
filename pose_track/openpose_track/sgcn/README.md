## 调用SGCN模型

```
# build necessities
cd ../graph/torchlight;
python setup.py install
```


```python
import sys
sys.path.insert(0, "/home/xyliu/cvToolBox/pose_track/lighttrack/sgcn")
import numpy as np
from sgcn import pose_matching  
sys.path.pop(0)
sys.path.pop(0)


B = [[[[  9.],[ 19.],[ 26.],[ 44.],[ 49.],[ 52.],[ 37.],[ 26.],[ 24.],[ 49.],[ 62.],[ 72.],[ 37.],[ 38.],[ 37.]]],[[[150.],[123.],[ 95.],[ 92.],[119.],[150.],[ 61.],[ 68.],[ 49.],[ 50.],[ 68.],[ 70.],[ 44.],[ 37.],[ 31.]]]]
A = [[[[ 28.],[ 56.],[ 73.],[ 57.],[ 74.],[ 78.],[ 55.],[ 84.],[ 88.],[ 59.],[ 50.],[ 43.],[ 74.],[ 74.],[ 74.]]],[[[143.],[129.],[ 91.],[ 88.],[120.],[156.],[ 67.],[ 66.],[ 47.],[ 45.],[ 68.],[ 70.],[ 37.],[ 27.],[ 18.]]]]

A=np.array(A)
B=np.array(B)
pose_matching(A,B)
```

## 速度
1.3ms每次 GPU 
0.9ms每次 CPU
5ms每次 API calling

## use gpu or not 
modify graph/gcn_utils/io.py L:108
