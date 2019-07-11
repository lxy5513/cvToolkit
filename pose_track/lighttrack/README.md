# pose track

## human bbox detector module
Yolov3

## pose module
simple-baseline or high-resolution network(hrnet)
```
test in 960*640 video  
the speed of handle each person  
simple-baseline: 90/second
hrnet:25/second
```  

## person association  
1. SpatialConsistency(bbox IoU)
2. pose simlarity(SGCN)

## demo  
python demo.py -p 0 (0->simple baseline; 1->hrnet)

## install 
pytorch 1.0.1+

### make  
`make gpu nms` 
