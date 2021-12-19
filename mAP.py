import os
from utility import *

np.set_printoptions(suppress=True)

names=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush']

truelabels = r'groundtruth'
predictpaths = r'predict'
save_dir = r'output'

def calculate(truelabel, predictpath):
    dictrue = []
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    plots = True
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    nc=len(names)
    confusion_matrix = ConfusionMatrix(nc)
    stats = []
    #loop of truelabels
    #遍历真实的标签
    for tl in os.listdir(truelabel):
        #read truelabel
        with open(os.path.join(truelabel, tl), 'r') as f:
            if not os.path.exists(os.path.join(predictpath, tl)):
                continue
            labels = []
            for fi in f.readlines():
                fi = fi.strip().split(',')
                classes, xmin, ymin, xmax, ymax = names.index(fi[0]), float(fi[1]), float(fi[2]), \
                                                            float(fi[3]), float(fi[4])
                labels.append([classes, xmin, ymin, xmax, ymax])
            labels = np.array(labels)
        
        #read corresponding predictions
        #读取相应的预测文件
        predn = []
        with open(os.path.join(predictpath, tl), 'r') as ff:
            for ffi in ff.readlines():
                ffi = ffi.strip().split(',')
                classes, xmin, ymin, xmax, ymax, score = names.index(ffi[0]), float(ffi[1]), float(ffi[2]), \
                                                            float(ffi[3]), float(ffi[4]), float(ffi[5])
                predn.append([xmin, ymin, xmax, ymax, score, classes])
         # sort prediction by score ascending
        #根据置信度从小到大排序
        predn = sorted(predn.__iter__(), key=lambda x:x[4], reverse=True)
        predn = np.array(predn)
        
        labels = torch.from_numpy(labels).to(device)
        predn = torch.from_numpy(predn).to(device)
        nl = len(labels)
        if len(predn) == 0:
            if nl:
                #for zeros prediction, we need to append blank lists
                #即使没有预测内容，仍然要添加相应的空列表
                stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
            continue
        if nl==0:
            continue
        detected = []
        #rows=len(predn), columns = 10,0.5:0.95, to save bool result large than iou threshold
        #(len(predn), 10),这里的10代表0.5:0.95的阈值，保存相应的布尔值，大于或者小于相应的iou阈值，从0.5~0.95
        correct = torch.zeros(predn.shape[0], niou, dtype=torch.bool, device=device)
        #truelabel class
        tcls = labels[:, 0].tolist()  # target class
        tcls_tensor = labels[:, 0]
        tbox = labels[:, 1:5]
        confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))
        #calculate iou of each class 
        #对于标签的每个类，计算相应的iou内容
        for cls in torch.unique(tcls_tensor):
            #get indexes of this class
            #得到该类对应的index
            ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
            pi = (cls == predn[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices
            # Search for detections
            if pi.shape[0]:
                # Prediction to target ious, for a prediction and one class, get maximum of iou value in all trues labels
                #每个prediction和ture label之间的iou最大值，和相应的标签index
                #predn[pi, :4] get special class, tbox[ti] get special class， max(1) get maximum value and index
                ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                # Append detections
                detected_set = set()
                count = 0
                #choose max iou>0.5' index
                #挑出最大的iou值大于0.5的index，也就是预测框
                for j in (ious > iouv[0]).nonzero(as_tuple=False):
                    d = ti[i[j]]  # detected target
                    if d.item() not in detected_set:
                        detected_set.add(d.item())
                        detected.append(d)
                        count += 1
                        #get iou threshold result of detection
                        #计算iou阈值相应的情况
                        correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                        if len(detected) == nl:  # all targets already located in image
                            break
        #collect all results
        #收集相应的计算内容
        stats.append((correct.cpu(), predn[:, 4].cpu(), predn[:, 5].cpu(), tcls))
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        #calculate ap value of each class
        #计算每个类相应的ap值
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    ffopen=open(os.path.join(save_dir, 'mAP.txt'), 'w')
    ffopen.write('predictpaths: '+predictpaths+'\n'+'truelabels: '+truelabels+'\n')
    s = ('%20s' + '%12s' * 5) % ('Class', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    print(s)
    # Print results
    pf = '%20s' + '%12.3g' * 5  # print format
    pff = '%20s' + '%16s' + '%12s'+ '%9s'+ '%12s'+ '%12s'
    ffopen.write(pff % ('classes', 'num truth', 'precision', 'recall', 'AP@0.5', 'AP@0.5:0.95')+'\n')
    ffopen.write('-'*100+'\n')
    ffopen.write(pf % ('all', nt.sum(), mp, mr, map50, map)+'\n')
    ffopen.write('-'*100+'\n')
    print(pf % ('all', nt.sum(), mp, mr, map50, map))
    #all         128         929       0.549       0.993        0.99       0.989

    # Print results per class
    # if (verbose or (nc <= 20 and not training)) and nc > 1 and len(stats):
    ap_dictionary = {}
    for i, c in enumerate(ap_class):
        print(pf % (names[c], nt[c], p[i], r[i], ap50[i], ap[i]))
        ap_dictionary[names[c]] = ap50[i]
        ffopen.write(pf % (names[c], nt[c], p[i], r[i], ap50[i], ap[i])+'\n')
        ffopen.write('-'*100+'\n')
    ffopen.close()
    confusion_matrix.plot(save_dir=save_dir, names=list(names))
    if plots:
        window_title = "mAP"
        plot_title = "mAP = {0:.2f}%".format(map50*100)
        x_label = "Average Precision"
        output_path = "mAP.png"
        to_show = True
        plot_color = 'royalblue'
        draw_plot_func(
            ap_dictionary,
            len(ap50),
            window_title,
            plot_title,
            x_label,
            os.path.join(save_dir, 'mAP.png'),
            to_show,
            plot_color,
            ""
            )

if __name__ == '__main__':
    calculate(truelabels, predictpaths)