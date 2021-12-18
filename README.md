## mAP
### Calculate mAP of detection result and groundtruth

`
pytorch=1.8.1
`

### Preparation
groundtruth dir: eachfile respond to a image with "label xmin ymin xmax ymax" <br>
predict dir:eachfile respond to a image with "label xmin ymin xmax ymax score" <br>

### Run
`
python mAP.py
`
<br>
                    <p>confusion matrix</p>                                           <p>precision_recall_curve</p>
<img src="output/mAP.png" width="39%" /><img src="output/precision_recall_curve.png" width="39%" />
### reference
[https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3)<br>
[https://github.com/Cartucho/mAP](https://github.com/Cartucho/mAP)