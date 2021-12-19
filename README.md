## mAP
### Calculate mAP of detection result and groundtruth

`
pytorch=1.8.1
`<br><br>
the groundtruth dir contains truth labels, the predict dir contains predicted labels.<br>

### Preparation
groundtruth dir: each file correspond to a special image with "label xmin ymin xmax ymax" <br>
predict dir: each file correspond to a special image with "label xmin ymin xmax ymax score" <br>
(xmin, ymin) is left-top coordinate, (xmax, ymax) is right-bottom coordinate<br>
<img src="output/coordi.png" width="39%" /><br>

### Run
`
python mAP.py
`<br>
<p>precision_recall_curve</p><br>
<img src="output/precision_recall_curve.png" width="80%" /><br>

### Process
<img src="output/example.png" width="80%" /><br>

### Reference <br>
[https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3)<br>
[https://github.com/Cartucho/mAP](https://github.com/Cartucho/mAP)