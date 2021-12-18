## mAP
### Calculate mAP of detection result and groundtruth

`
pytorch=1.8.1
`<br>
the groundtruth dir contains truth labels, the predict dir contains predicted labels.<br>
### Preparation
groundtruth dir: eachfile respond to a image with "label xmin ymin xmax ymax" <br>
predict dir:eachfile respond to a image with "label xmin ymin xmax ymax score" <br>
(xmin, ymin) is left-top coordinate, (xmax, ymax) is right-bottom coordinate<br>
<img src="output/coordi.png" width="39%" /><br>
### Run
`
python mAP.py
`
<br>
<p>precision_recall_curve</p><br>
<img src="output/precision_recall_curve.png" width="80%" /><br>
 <p>confusion matrix</p><br>
<img src="output/mAP.png" width="80%" />
### reference
[https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3)<br>
[https://github.com/Cartucho/mAP](https://github.com/Cartucho/mAP)