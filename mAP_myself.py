import os
from utility import *

def calculate(truelabel, predictpath, min_confidence):
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    plots = True
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    nc=len(names)
    confusion_matrix = ConfusionMatrix(nc)
    collect = []
    #loop of truelabels
    #遍历真实的标签
    for tl in os.listdir(truelabel):
        #read truelabel
        labels = []
        with open(os.path.join(truelabel, tl), 'r') as f:
            for fi in f.readlines():
                fi = fi.strip().split(',')
                classes, xmin, ymin, xmax, ymax = names.index(fi[0]), float(fi[1]), float(fi[2]), \
                                                            float(fi[3]), float(fi[4])
                labels.append([xmin, ymin, xmax, ymax, classes])
            labels = np.array(labels)
        
        #read corresponding predictions
        #读取相应的预测文件
        predn = []
        if os.path.exists(os.path.join(predictpath, tl)):
            with open(os.path.join(predictpath, tl), 'r') as ff:
                for ffi in ff.readlines():
                    ffi = ffi.strip().split(',')
                    classes, xmin, ymin, xmax, ymax, score = names.index(ffi[0]), float(ffi[1]), float(ffi[2]), \
                                                                float(ffi[3]), float(ffi[4]), float(ffi[5])
                    if score < min_confidence:
                        continue
                    predn.append([xmin, ymin, xmax, ymax, score, classes])
        else:
            pass
        if len(predn) == 0 and len(labels) > 0:
            tcls = labels[:, -1].tolist()  # target class
            collect.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
            continue
        if len(labels)==0:
            continue
        #rows=len(predn), columns = 10,0.5:0.95, to save bool result large than iou threshold
        #(len(predn), 10),这里的10代表0.5:0.95的阈值，保存相应的布尔值，大于或者小于相应的iou阈值，从0.5~0.95
        correct = torch.zeros(len(predn), niou, dtype=torch.bool, device=device)
        tcls = labels[:, -1].tolist()  # target class

         # sort prediction by score ascending
        #根据置信度从小到大排序
        predn = sorted(predn.__iter__(), key=lambda x:x[4], reverse=True)
        predn = np.array(predn)
        unique_class = np.unique(predn[:, -1])
        detected = []
        tbox = labels[:, 1:5]
        # confusion_matrix.process_batch(torch.from_numpy(predn), torch.cat((torch.from_numpy(labels[:, 0:1]), torch.from_numpy(tbox)), 1))
        #对于标签的每个类，计算相应的iou内容
        for uqc in unique_class:
            pred_index = np.where(predn[:, -1]==uqc)[0]
            true_index = np.where(labels[:, -1]==uqc)[0]
            now_predict = predn[pred_index]
            now_trueth  = labels[true_index]
            if len(now_trueth)==0:
                continue
            iou_result = iou_calculate(now_trueth[:, :4], now_predict[:, :4]) #(len(now_predict),  len(now_trueth)
            #每个prediction和所有ture label之间的iou最大值，和相应的标签index
            iou, ind = torch.max(iou_result, dim=1)
            #挑出最大的iou值大于0.5的index，也就是预测框
            detected_set = set()
            for j in torch.where(iou > iouv[0])[0]:         # 预测框 iou>0.5
                findone = true_index[ind[j]]            #拿到iou最大对应的truelabel的index
                if findone.item() not in detected_set:  #一个真实框只需要一个predict，所以其他的max_iou > 0.5的预测框都会被判定负样本
                    detected_set.add(findone.item())    #第findone个真实框已经找到iou>0.5的预测框，加入列表里
                    detected.append(findone)
                    correct[pred_index[j]] = iou[j] > iouv  # iou_thres is 1xn
                    if len(detected)==len(labels):      #每个真实框都找到了相应的predict
                        break
        collect.append([correct, predn[:, 4], predn[:, 5], tcls]) #mask, score, classes, truth_class
    collect = np.array(collect)
    collect = [ np.concatenate(collect[:, 0], 0),   
                np.concatenate(collect[:, 1], 0),
                np.concatenate(collect[:, 2], 0),
                np.concatenate(collect[:, 3], 0),
               ]
    if len(collect) > 0 and collect[0].any():
        #calculate ap value of each class
        #计算每个类相应的ap值
        p, r, ap, f1, ap_class = ap_per_class(*collect, plot=plots, save_dir=save_dir, names=names)
        p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(collect[3].astype(np.int64), minlength=nc)  # number of targets per class
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
    # confusion_matrix.plot(save_dir=save_dir, names=list(names))
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
    min_confidence = 0.01

    calculate(truelabels, predictpaths, min_confidence = min_confidence)