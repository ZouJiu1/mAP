## mAP
### update-2023
I rewrite some codes, It can be more better to understand, so should use `mAP_myself.py` this file to calculate or understand<br>
There is a new parameter `min_confidence` to ignore those prediction below confidence threshold.<br>
### Calculate mAP of detection result and groundtruth

`
pytorch=1.8.1
`<br><br>
the groundtruth dir contains truth labels, the predict dir contains predicted labels.<br>

### Preparation
groundtruth dir: each file correspond to a special image with "label xmin ymin xmax ymax" <br>
predict dir: each file correspond to a special image with "label xmin ymin xmax ymax score" <br>
(xmin, ymin) is left-top coordinate, (xmax, ymax) is right-bottom coordinate<br>
<img src="output/coordi.jpg" width="39%" /><br>

### Run
`
python mAP.py
`<br>
<p>precision_recall_curve</p><br>
<img src="output/precision_recall_curve.png" width="80%" /><br>

### Process
<img src="output/example.png" width="80%" /><br>

### Calculating mAP example
turelabels<br>
<img src="output/turelabel.png" width="27%" /><br>

prediction<br>
<img src="output/predict.png" width="27%" /><br>

calculation<br>
<img src="output/calculateAP.png" width="98%" /><br>

results<br>
<img src="output/class0AP.png" width="32%" />  <img src="output/class1AP.png" width="32%" />  <img src="output/class2AP.png" width="32%" /><br>

### Reference <br>
[https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3)<br>
[https://github.com/Cartucho/mAP](https://github.com/Cartucho/mAP)

### Mainpage <br>
[https://blog.csdn.net/m0_50617544/article/details/121893818](https://blog.csdn.net/m0_50617544/article/details/121893818)<br>
[https://zhuanlan.zhihu.com/p/449822471](https://zhuanlan.zhihu.com/p/449822471)