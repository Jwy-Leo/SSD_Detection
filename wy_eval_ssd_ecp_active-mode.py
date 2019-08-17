import torch
from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.datasets.voc_dataset import VOCDataset
from vision.datasets.open_images import OpenImagesDataset
from vision.datasets.coco_dataset import CocoDetection
from vision.datasets.EuroCity_dataset import EuroCity_Dataset
from vision.datasets.folder_dataset import FolderDataset
from vision.utils import box_utils, measurements
from vision.utils.misc import str2bool, Timer
import argparse
import pathlib
import numpy as np
import logging
import sys
import os
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from detection_configuration import load_model_configuration
from operator import itemgetter
def Arguments():
    parser = argparse.ArgumentParser(description="SSD Evaluation on VOC Dataset.")
    parser.add_argument('--net', default="vgg16-ssd",
                        help="The network architecture, it should be of mb1-ssd, mb1-ssd-lite, mb2-ssd-lite or vgg16-ssd.")
    parser.add_argument("--trained_model_root_path", type=str)

    parser.add_argument("--dataset_type", default="voc", type=str,
                        help='Specify dataset type. Currently support voc and open_images.')
    parser.add_argument("--dataset", type=str, help="The root directory of the VOC dataset or Open Images dataset or coco dataset or ecp dataset.")
    parser.add_argument("--label_file", type=str, help="The label file path.")
    parser.add_argument("--use_cuda", type=str2bool, default=True)
    parser.add_argument("--use_2007_metric", type=str2bool, default=True)
    parser.add_argument("--nms_method", type=str, default="hard")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="The threshold of Intersection over Union.")
    parser.add_argument("--eval_dir", default="../experiments/eval_results", type=str, help="The directory to store evaluation results.")
    parser.add_argument('--mb2_width_mult', default=1.0, type=float,
                        help='Width Multiplifier for MobilenetV2')
    parser.add_argument('--config', default='config/default_setting.yaml', type=str, help='Configuration')
    args = parser.parse_args()
    print(args)
    configuration = load_model_configuration(args.config)
    configuration["flow_control"] = {}
    variable_dict = vars(args)
    for key in variable_dict.keys():
        configuration['flow_control'][key] = variable_dict[key]
    return configuration
def main_single_model(args):
    _temp_str = str.split(args['flow_control']['trained_model_root_path'],"/")
    if _temp_str[-1]!="":
        _temp_str[-1] += "_eval_results"
    else:
        _temp_str[-2] += "_eval_results"
    log_path = os.path.join("/".join(_temp_str),'logs')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args['flow_control']['use_cuda'] else "cpu")

    # eval_path = pathlib.Path(args.eval_dir)
    # eval_path.mkdir(exist_ok=True)
    if not os.path.exists(args['flow_control']['eval_dir']):
        os.mkdir(args['flow_control']['eval_dir'])
    timer = Timer()
    class_names = [name.strip() for name in open(args['flow_control']['label_file']).readlines()]


    _net = args['flow_control']['net']
    _dataset_type = args['flow_control']['dataset_type']

    if _dataset_type == "voc":
        raise NotImplementedError("Not implement error")
        dataset = VOCDataset(args['flow_control']['dataset'], is_test=True)
    elif _dataset_type == 'open_images':
        raise NotImplementedError("Not implement error")
        dataset = OpenImagesDataset(args['flow_control']['dataset'], dataset_type="test")
    elif _dataset_type == "coco":
        # dataset = CocoDetection("/home/wenyen4desh/datasets/coco/test2017","/home/wenyen4desh/datasets/annotations/image_info_test2017.json") 
        #dataset = CocoDetection("../../dataset/datasets/coco/val2017","../../dataset/datasets/coco/annotations/instances_val2017.json") 
        # dataset = CocoDetection("/home/wenyen4desh/datasets/coco/train2017","/home/wenyen4desh/datasets/coco/annotations/instances_train2017.json") 
        dataset = CocoDetection(args['Datasets']['coco']['val_image_path'],args['Datasets']['coco']['val_anno_path']) 
    elif _dataset_type == "ecp":
        dataset = EuroCity_Dataset(args['Datasets']['ecp']['val_image_path'],args['Datasets']['ecp']['val_anno_path']) 
    elif _dataset_type == "folder":        
        dataset = FolderDataset(args['Datasets']['folder']['val_image_path'])
    true_case_stat, all_gb_boxes, all_difficult_cases = group_annotation_by_class(dataset)
    if _net == 'vgg16-ssd':
        net = create_vgg_ssd(len(class_names), is_test=True)
    elif _net == 'mb1-ssd':
        net = create_mobilenetv1_ssd(len(class_names), is_test=True)
    elif _net == 'mb1-ssd-lite':
        net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
    elif _net == 'sq-ssd-lite':
        net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
    elif _net == 'mb2-ssd-lite':
        net = create_mobilenetv2_ssd_lite(len(class_names), width_mult=args['flow_control']['mb2_width_mult'], is_test=True)
    else:
        logging.fatal("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        parser.print_help(sys.stderr)
        sys.exit(1)  

    #train_transform = MatchPrior(config.priors, config.center_variance,
    #                              config.size_variance, 0.5)

    #test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)
    # import pdb;pdb.set_trace()
    import pdb;pdb.set_trace()
    RootPath, _folders, _files = next(iter(os.walk(args['flow_control']['trained_model_root_path'])))
    _folders = sorted(_folders)
    # for _folder in _folders:
    _folder = _folders[0]
    Log_File = open(os.path.join(log_path,_folder+".log"),'w')
    # try:
    active_folder = os.path.join(RootPath, _folder)
    _MRP, _, _files = next(iter(os.walk(active_folder)))
    compare_item = [ float(str.split(_file[:-4],"-")[-1]) for _file in _files]
    _files_tuple = zip(_files,compare_item)
    _files_tuple = [_temp_f_t for _temp_f_t in _files_tuple]
    _files_tuple.sort(key=itemgetter(1))
    Target_model = _files_tuple[0][0]
    Target_model_path = os.path.join(active_folder,Target_model)
    
    ########################## automatically validation ############################################
    timer.start("Load Model")
    net.load(Target_model_path)
    net = net.to(DEVICE)
    Log_File.write('It took {} seconds to load the model.\n'.format(timer.end("Load Model")))
    # print('It took {} seconds to load the model.'.format(timer.end("Load Model")))
    _nms_method = args['flow_control']['nms_method']
    if _net == 'vgg16-ssd':
        predictor = create_vgg_ssd_predictor(net, nms_method=_nms_method, device=DEVICE)
    elif _net == 'mb1-ssd':
        predictor = create_mobilenetv1_ssd_predictor(net, nms_method=_nms_method, device=DEVICE)
    elif _net == 'mb1-ssd-lite':
        predictor = create_mobilenetv1_ssd_lite_predictor(net, nms_method=_nms_method, device=DEVICE)
    elif _net == 'sq-ssd-lite':
        predictor = create_squeezenet_ssd_lite_predictor(net,nms_method=_nms_method, device=DEVICE)
    elif _net == 'mb2-ssd-lite':
        predictor = create_mobilenetv2_ssd_lite_predictor(net, nms_method=_nms_method, device=DEVICE)
    else:
        logging.fatal("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    results = []  
    # Predict Bounding Box
    for i in range(len(dataset)):
        # print("process image", i)
        Log_File.write("process image {}\n".format(i))
        timer.start("Load Image")
        image = dataset.get_image(i)
        # print("Load Image: {:4f} seconds.".format(timer.end("Load Image")))
        Log_File.write("Load Image: {:4f} seconds.\n".format(timer.end("Load Image")))
        timer.start("Predict")
        boxes, labels, probs = predictor.predict(image)
        # print("Prediction: {:4f} seconds.".format(timer.end("Predict")))
        Log_File.write("Prediction: {:4f} seconds.\n".format(timer.end("Predict")))
        indexes = torch.ones(labels.size(0), 1, dtype=torch.float32) * i
        results.append(torch.cat([
            indexes.reshape(-1, 1),
            labels.reshape(-1, 1).float(),
            probs.reshape(-1, 1),
            boxes + 1.0  # matlab's indexes start from 1
        ], dim=1))
    results = torch.cat(results)
    
    _list_active_folder = str.split(active_folder,"/")
    # _list_active_folder.insert(-1,'eval_results')
    _list_active_folder[-2] = _list_active_folder[-2] + "_eval_results"
    _prediction_path = "/".join(_list_active_folder)
    if not os.path.exists(_prediction_path):
        os.makedirs(_prediction_path)
    # Write the result to file 
    for class_index, class_name in enumerate(class_names):
        if class_index == 0: continue  # ignore background
        file_name = "det_test_{}.txt".format(class_name)
        prediction_path = os.path.join(_prediction_path,file_name) 
        with open(prediction_path, "w") as f:
            sub = results[results[:, 1] == class_index, :]
            for i in range(sub.size(0)):
                prob_box = sub[i, 2:].numpy()
                image_id,_ = dataset.get_annotation(int(sub[i, 0]))
                f.write(str(image_id) + " " + " ".join([str(v) for v in prob_box])+"\n")
                # image_id = dataset.ids[int(sub[i, 0])]
                # print(str(image_id) + " " + " ".join([str(v) for v in prob_box]), file=f)
    
    aps = []
    prcs = []
    recalls = []
    Log_File.write("\n\nAverage Precision Per-class:\n")
    # print("\n\nAverage Precision Per-class:")
    # for class_index, class_name in enumerate(class_names):
    for class_index in true_case_stat.keys():
        class_name = class_names[class_index]
        # if class_index not in true_case_stat.keys():
        #     Log_File.write("{}: 0.0\n".format(class_name))
        if class_index == 0:
            continue
        file_name = "det_test_{}.txt".format(class_name)
        prediction_path = os.path.join(_prediction_path,file_name) 
        # Pascal@0.5 evaluation method 
        ap, precision, recall = compute_average_precision_per_class(
            args,
            true_case_stat[class_index],
            all_gb_boxes[class_index],
            all_difficult_cases[class_index],
            prediction_path,
            args['flow_control']['iou_threshold'],
            args['flow_control']['use_2007_metric']
        )

        # # COCO eval 

        # ap, precision, recall = coco_ap_per_class(
        #     true_case_stat[class_index],
        #     all_gb_boxes[class_index],
        #     all_difficult_cases[class_index],
        #     prediction_path,
        #     args.use_2007_metric
        # )

        aps.append(ap)
        prcs.append(precision)
        recalls.append(recall)
        Log_File.write("{}: {}\n".format(class_name,ap))
        # print("{}: {}".format(class_name,ap))
    Log_File.write("\nAverage Precision Across All Classes:{}\n".format(sum(aps[0:5])/len(aps[0:5])))
    Log_File.write("\nAverage Precision :{}\n".format(sum(prcs[0:5])/len(prcs[0:5])))
    Log_File.write("\nAverage Recall :{}\n".format(sum(recalls[0:5])/len(recalls[0:5])))
    Log_File.close()
def main(args):
    _temp_str = str.split(args['flow_control']['trained_model_root_path'],"/")
    if _temp_str[-1]!="":
        _temp_str[-1] += "_eval_results"
    else:
        _temp_str[-2] += "_eval_results"
    log_path = os.path.join("/".join(_temp_str),'logs')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args['flow_control']['use_cuda'] else "cpu")

    # eval_path = pathlib.Path(args.eval_dir)
    # eval_path.mkdir(exist_ok=True)
    if not os.path.exists(args['flow_control']['eval_dir']):
        os.mkdir(args['flow_control']['eval_dir'])
    timer = Timer()
    class_names = [name.strip() for name in open(args['flow_control']['label_file']).readlines()]


    _net = args['flow_control']['net']
    _dataset_type = args['flow_control']['dataset_type']

    if _dataset_type == "voc":
        raise NotImplementedError("Not implement error")
        dataset = VOCDataset(args['flow_control']['dataset'], is_test=True)
    elif _dataset_type == 'open_images':
        raise NotImplementedError("Not implement error")
        dataset = OpenImagesDataset(args['flow_control']['dataset'], dataset_type="test")
    elif _dataset_type == "coco":
        # dataset = CocoDetection("/home/wenyen4desh/datasets/coco/test2017","/home/wenyen4desh/datasets/annotations/image_info_test2017.json") 
        #dataset = CocoDetection("../../dataset/datasets/coco/val2017","../../dataset/datasets/coco/annotations/instances_val2017.json") 
        # dataset = CocoDetection("/home/wenyen4desh/datasets/coco/train2017","/home/wenyen4desh/datasets/coco/annotations/instances_train2017.json") 
        dataset = CocoDetection(args['Datasets']['coco']['val_image_path'],args['Datasets']['coco']['val_anno_path']) 
    elif _dataset_type == "ecp":
        dataset = EuroCity_Dataset(args['Datasets']['ecp']['val_image_path'],args['Datasets']['ecp']['val_anno_path']) 
    elif _dataset_type == "folder":        
        dataset = FolderDataset(args['Datasets']['folder']['val_image_path'])
    true_case_stat, all_gb_boxes, all_difficult_cases = group_annotation_by_class(dataset)
    if _net == 'vgg16-ssd':
        net = create_vgg_ssd(len(class_names), is_test=True)
    elif _net == 'mb1-ssd':
        net = create_mobilenetv1_ssd(len(class_names), is_test=True)
    elif _net == 'mb1-ssd-lite':
        net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
    elif _net == 'sq-ssd-lite':
        net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
    elif _net == 'mb2-ssd-lite':
        net = create_mobilenetv2_ssd_lite(len(class_names), width_mult=args['flow_control']['mb2_width_mult'], is_test=True)
    else:
        logging.fatal("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        parser.print_help(sys.stderr)
        sys.exit(1)  

    #train_transform = MatchPrior(config.priors, config.center_variance,
    #                              config.size_variance, 0.5)

    #test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)
    # import pdb;pdb.set_trace()
    import pdb;pdb.set_trace()
    RootPath, _folders, _files = next(iter(os.walk(args['flow_control']['trained_model_root_path'])))
    _folders = sorted(_folders)
    for _folder in _folders:
        Log_File = open(os.path.join(log_path,_folder+".log"),'w')
        # try:
        active_folder = os.path.join(RootPath, _folder)
        _MRP, _, _files = next(iter(os.walk(active_folder)))
        compare_item = [ float(str.split(_file[:-4],"-")[-1]) for _file in _files]
        _files_tuple = zip(_files,compare_item)
        _files_tuple = [_temp_f_t for _temp_f_t in _files_tuple]
        _files_tuple.sort(key=itemgetter(1))
        Target_model = _files_tuple[0][0]
        Target_model_path = os.path.join(active_folder,Target_model)
        
        ########################## automatically validation ############################################
        timer.start("Load Model")
        net.load(Target_model_path)
        net = net.to(DEVICE)
        Log_File.write('It took {} seconds to load the model.\n'.format(timer.end("Load Model")))
        # print('It took {} seconds to load the model.'.format(timer.end("Load Model")))
        _nms_method = args['flow_control']['nms_method']
        if _net == 'vgg16-ssd':
            predictor = create_vgg_ssd_predictor(net, nms_method=_nms_method, device=DEVICE)
        elif _net == 'mb1-ssd':
            predictor = create_mobilenetv1_ssd_predictor(net, nms_method=_nms_method, device=DEVICE)
        elif _net == 'mb1-ssd-lite':
            predictor = create_mobilenetv1_ssd_lite_predictor(net, nms_method=_nms_method, device=DEVICE)
        elif _net == 'sq-ssd-lite':
            predictor = create_squeezenet_ssd_lite_predictor(net,nms_method=_nms_method, device=DEVICE)
        elif _net == 'mb2-ssd-lite':
            predictor = create_mobilenetv2_ssd_lite_predictor(net, nms_method=_nms_method, device=DEVICE)
        else:
            logging.fatal("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
            parser.print_help(sys.stderr)
            sys.exit(1)
    
        results = []  
        # Predict Bounding Box
        for i in range(len(dataset)):
            # print("process image", i)
            Log_File.write("process image {}\n".format(i))
            timer.start("Load Image")
            image = dataset.get_image(i)
            # print("Load Image: {:4f} seconds.".format(timer.end("Load Image")))
            Log_File.write("Load Image: {:4f} seconds.\n".format(timer.end("Load Image")))
            timer.start("Predict")
            boxes, labels, probs = predictor.predict(image)
            # print("Prediction: {:4f} seconds.".format(timer.end("Predict")))
            Log_File.write("Prediction: {:4f} seconds.\n".format(timer.end("Predict")))
            indexes = torch.ones(labels.size(0), 1, dtype=torch.float32) * i
            results.append(torch.cat([
                indexes.reshape(-1, 1),
                labels.reshape(-1, 1).float(),
                probs.reshape(-1, 1),
                boxes + 1.0  # matlab's indexes start from 1
            ], dim=1))
        results = torch.cat(results)
        
        _list_active_folder = str.split(active_folder,"/")
        # _list_active_folder.insert(-1,'eval_results')
        _list_active_folder[-2] = _list_active_folder[-2] + "_eval_results"
        _prediction_path = "/".join(_list_active_folder)
        if not os.path.exists(_prediction_path):
            os.makedirs(_prediction_path)
        # Write the result to file 
        for class_index, class_name in enumerate(class_names):
            if class_index == 0: continue  # ignore background
            file_name = "det_test_{}.txt".format(class_name)
            prediction_path = os.path.join(_prediction_path,file_name) 
            with open(prediction_path, "w") as f:
                sub = results[results[:, 1] == class_index, :]
                for i in range(sub.size(0)):
                    prob_box = sub[i, 2:].numpy()
                    image_id,_ = dataset.get_annotation(int(sub[i, 0]))
                    f.write(str(image_id) + " " + " ".join([str(v) for v in prob_box])+"\n")
                    # image_id = dataset.ids[int(sub[i, 0])]
                    # print(str(image_id) + " " + " ".join([str(v) for v in prob_box]), file=f)
        
        aps = []
        prcs = []
        recalls = []
        Log_File.write("\n\nAverage Precision Per-class:\n")
        # print("\n\nAverage Precision Per-class:")
        # for class_index, class_name in enumerate(class_names):
        for class_index in true_case_stat.keys():
            class_name = class_names[class_index]
            # if class_index not in true_case_stat.keys():
            #     Log_File.write("{}: 0.0\n".format(class_name))
            if class_index == 0:
                continue
            file_name = "det_test_{}.txt".format(class_name)
            prediction_path = os.path.join(_prediction_path,file_name) 
            # Pascal@0.5 evaluation method 
            ap, precision, recall = compute_average_precision_per_class(
                args,
                true_case_stat[class_index],
                all_gb_boxes[class_index],
                all_difficult_cases[class_index],
                prediction_path,
                args['flow_control']['iou_threshold'],
                args['flow_control']['use_2007_metric']
            )

            # # COCO eval 

            # ap, precision, recall = coco_ap_per_class(
            #     true_case_stat[class_index],
            #     all_gb_boxes[class_index],
            #     all_difficult_cases[class_index],
            #     prediction_path,
            #     args.use_2007_metric
            # )

            aps.append(ap)
            prcs.append(precision)
            recalls.append(recall)
            Log_File.write("{}: {}\n".format(class_name,ap))
            # print("{}: {}".format(class_name,ap))
        Log_File.write("\nAverage Precision Across All Classes:{}\n".format(sum(aps[0:5])/len(aps[0:5])))
        Log_File.write("\nAverage Precision :{}\n".format(sum(prcs[0:5])/len(prcs[0:5])))
        Log_File.write("\nAverage Recall :{}\n".format(sum(recalls[0:5])/len(recalls[0:5])))
        Log_File.close()
        # print("\nAverage Precision Across All Classes:{}".format(sum(aps[0:5])/len(aps[0:5])))
        # print("\nAverage Precision :{}".format(sum(prcs[0:5])/len(prcs[0:5])))
        # print("\nAverage Recall :{}".format(sum(recalls[0:5])/len(recalls[0:5])))
        # except:
        #     Log_File.close()

def group_annotation_by_class(dataset):
    true_case_stat = {}
    all_gt_boxes = {}
    all_difficult_cases = {}
    set_class = []
    for i in range(len(dataset)):
        image_id, annotation = dataset.get_annotation(i)
        gt_boxes, classes, is_difficult = annotation
        set_class.extend(classes.tolist())
        gt_boxes = torch.from_numpy(gt_boxes)
        for i, difficult in enumerate(is_difficult):
            class_index = int(classes[i])
            gt_box = gt_boxes[i]
            if not difficult:
                true_case_stat[class_index] = true_case_stat.get(class_index, 0) + 1

            if class_index not in all_gt_boxes:
                all_gt_boxes[class_index] = {}
            if image_id not in all_gt_boxes[class_index]:
                all_gt_boxes[class_index][image_id] = []
            all_gt_boxes[class_index][image_id].append(gt_box)
            if class_index not in all_difficult_cases:
                all_difficult_cases[class_index]={}
            if image_id not in all_difficult_cases[class_index]:
                all_difficult_cases[class_index][image_id] = []
            all_difficult_cases[class_index][image_id].append(difficult)

    for class_index in all_gt_boxes:
        for image_id in all_gt_boxes[class_index]:
            all_gt_boxes[class_index][image_id] = torch.stack(all_gt_boxes[class_index][image_id])
    for class_index in all_difficult_cases:
        for image_id in all_difficult_cases[class_index]:
            all_gt_boxes[class_index][image_id] = torch.tensor(all_gt_boxes[class_index][image_id])
    return true_case_stat, all_gt_boxes, all_difficult_cases

def coco_ap_per_class(num_true_cases, gt_boxes, difficult_cases,
                      prediction_file, use_2007_metric):
    aps, precs, recalls = [],[],[]
    _measurement_func = measurements.compute_voc2007_average_precision if use_2007_metric else measurements.compute_average_precision
    with open(prediction_file) as f:
        image_ids = []
        boxes = []
        scores = []
        for line in f:
            t = line.rstrip().split(" ")
            image_ids.append(t[0])
            scores.append(float(t[1]))
            box = torch.tensor([float(v) for v in t[2:]]).unsqueeze(0)
            box -= 1.0  # convert to python format where indexes start from 0
            boxes.append(box)
        scores = np.array(scores)
        sorted_indexes = np.argsort(-scores)
        boxes = [boxes[i] for i in sorted_indexes]
        image_ids = [image_ids[i] for i in sorted_indexes]
        
    max_ious = []
    max_args = []
    for i, image_id in enumerate(image_ids):
        box = boxes[i]
        image_id = int(image_id)
        # if image_id not in gt_boxes.keys():
        #     false_positive[i] = 1
        #     continue
        if image_id in gt_boxes.keys():
            gt_box = gt_boxes[image_id]
            ious = box_utils.iou_of(box, gt_box)
            max_iou = torch.max(ious).item() 
            max_arg = torch.argmax(ious).item()
        else:
            max_iou = 0.0
            max_arg = 0
        max_ious.append(max_iou)
        max_args.append(max_arg)
    for _iou_thresh  in [0.5+0.05 * i for i in range(0,10)]:
        true_positive = np.zeros(len(image_ids))
        false_positive = np.zeros(len(image_ids))
        matched = set()
        for i,(max_iou,max_arg,image_id) in enumerate(zip(max_ious,max_args,image_ids)):
            image_id = int(image_id)
            if max_iou > _iou_thresh:
                if difficult_cases[image_id][max_arg] == 0:
                    if (image_id, max_arg) not in matched:
                        true_positive[i] = 1
                        matched.add((image_id, max_arg))
                    else:
                        false_positive[i] = 1
            else:
                false_positive[i] = 1

        true_positive = true_positive.cumsum()
        false_positive = false_positive.cumsum()
        precision = true_positive / (true_positive + false_positive + 1e-8 )
        recall = true_positive / num_true_cases

        _vtp = true_positive[-1] if len(true_positive)!=0 else 0
        _vfp = false_positive[-1] if len(false_positive)!=0 else 0
        ap = _measurement_func(precision,recall)
        _prec = _vtp / (_vtp+ _vfp + 1e-8)
        _recall =  _vtp/num_true_cases
        aps.append(ap)
        precs.append(_prec)
        recalls.append(_recall)
    return sum(aps)/len(aps), sum(precs)/len(precs), sum(recalls) / len(recalls)
def compute_average_precision_per_class(args, num_true_cases, gt_boxes, difficult_cases,
                                        prediction_file, iou_threshold, use_2007_metric):
    with open(prediction_file) as f:
        image_ids = []
        boxes = []
        scores = []
        for line in f:
            t = line.rstrip().split(" ")
            image_ids.append(t[0])
            scores.append(float(t[1]))
            box = torch.tensor([float(v) for v in t[2:]]).unsqueeze(0)
            box -= 1.0  # convert to python format where indexes start from 0
            boxes.append(box)
        scores = np.array(scores)
        sorted_indexes = np.argsort(-scores)
        boxes = [boxes[i] for i in sorted_indexes]
        image_ids = [image_ids[i] for i in sorted_indexes]
        true_positive = np.zeros(len(image_ids))
        false_positive = np.zeros(len(image_ids))
        matched = set()
        for i, image_id in enumerate(image_ids):
            box = boxes[i]
            
            if args['flow_control']['dataset_type']=="voc":
                pass
            else:
                image_id = int(image_id)
                
            if image_id not in gt_boxes.keys():
                false_positive[i] = 1
                continue
            
            gt_box = gt_boxes[image_id]
            ious = box_utils.iou_of(box, gt_box)
            max_iou = torch.max(ious).item()
            max_arg = torch.argmax(ious).item()
            if max_iou > iou_threshold:
                if difficult_cases[image_id][max_arg] == 0:
                    if (image_id, max_arg) not in matched:
                        true_positive[i] = 1
                        matched.add((image_id, max_arg))
                    else:
                        false_positive[i] = 1
            else:
                false_positive[i] = 1

    true_positive = true_positive.cumsum()
    false_positive = false_positive.cumsum()
    precision = true_positive / (true_positive + false_positive + 1e-8 )
    recall = true_positive / num_true_cases
    # import pdb;pdb.set_trace()
    _vtp = true_positive[-1] if len(true_positive)!=0 else 0
    _vfp = false_positive[-1] if len(false_positive)!=0 else 0
    if use_2007_metric:
        return measurements.compute_voc2007_average_precision(precision, recall), _vtp / (_vtp+ _vfp + 1e-8), _vtp/num_true_cases
    else:
        return measurements.compute_average_precision(precision, recall)


if __name__ == '__main__':
    args = Arguments()
    # main(args)
    main_single_model(args)



