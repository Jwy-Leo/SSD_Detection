import torch
from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.datasets.voc_dataset import VOCDataset
from vision.datasets.open_images import OpenImagesDataset
from vision.datasets.coco_dataset import CocoDetection
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
import cv2
def eval_whole_image(dataset, stride, predictor):
    for i in range(0,len(dataset),stride):
        image = dataset.get_image(i)
        gt_bbox, gt_label = dataset.get_annotation(i)
        gt_bbox, gt_label = gt_label
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        for box, _label in zip(gt_bbox,gt_label):
            if _label ==2 or _label==3:
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 4)
        # Whole image
        boxes, labels, probs = predictor.predict(image,10,0.5)
        boxes,labels,probs = boxes.data.numpy(), labels.data.numpy(), probs.data.numpy()
        for box, _label, _prob in zip(boxes,labels,probs):
            #if _prob < 0.7: continue
            
            box = box.astype(int)            
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
            cv2.putText(image, str(_label), (box[0]+20, box[1]+40),cv2.FONT_HERSHEY_SIMPLEX,
                1,  # font scale
                (255, 0, 255),
                2)  # line type
            
            cv2.imshow('annotated', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
def eval_subblock_image(dataset, stride, predictor):
    import itertools
    input_size = 300.
    for i in range(0,len(dataset), stride):
        image = dataset.get_image(i)
        gt_bbox, gt_label = dataset.get_annotation(i)
        gt_bbox, gt_label = gt_label
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        for box, _label in zip(gt_bbox,gt_label):
            if _label ==2 or _label==3:
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 4)

        box_coll, label_coll, probs_coll = [], [], []
        w_iter_time = int(float(image.shape[0])/input_size) + 1
        h_iter_time = int(float(image.shape[1])/input_size) + 1
        for j, i in itertools.product(range(w_iter_time), range(h_iter_time)):
            if j != w_iter_time - 1 :
                h_min_size, h_max_size = int(j * input_size), int(min((j+1) * input_size, image.shape[0]))
            else:
                h_min_size, h_max_size = int(image.shape[0] - input_size), image.shape[0]
            if i != h_iter_time - 1 :
                w_min_size, w_max_size = int(i * input_size), int(min((i+1) * input_size, image.shape[1]))
            else:
                w_min_size, w_max_size = int(image.shape[1] - input_size), image.shape[1]
            sub_image = image[h_min_size:h_max_size, w_min_size:w_max_size, :]
            boxes, labels, probs = predictor.predict(sub_image,10,0.5)
            boxes,labels,probs = boxes.data.numpy(), labels.data.numpy(), probs.data.numpy()
            if boxes.shape[0]!=0:
                boxes[:,0:4:2] += w_min_size
                boxes[:,1:4:2] += h_min_size
                box_coll.append(boxes)
                label_coll.append(labels)
                probs_coll.append(probs)
        if len(box_coll)==0:
            continue
        boxes = np.vstack(box_coll)
        labels = np.hstack(label_coll)
        probs = np.hstack(probs_coll)
        for box, _label, _prob in zip(boxes,labels,probs):
            #if _prob < 0.7: continue
            
            box = box.astype(int)            
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
            cv2.putText(image, str(_label), (box[0]+20, box[1]+40),cv2.FONT_HERSHEY_SIMPLEX,
                1,  # font scale
                (255, 0, 255),
                2)  # line type
            
            cv2.imshow('annotated', image)
            
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
def main(args):
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")
    if not os.path.exists(args.eval_dir):
        os.mkdir(args.eval_dir)
    timer = Timer()
    class_names = [name.strip() for name in open(args.label_file).readlines()]


    # dataset = Folder_image_set()

    # true_case_stat, all_gb_boxes, all_difficult_cases = group_annotation_by_class(dataset)
    if args.net == 'vgg16-ssd':
        net = create_vgg_ssd(len(class_names), is_test=True)
    elif args.net == 'mb1-ssd':
        net = create_mobilenetv1_ssd(len(class_names), is_test=True)
    elif args.net == 'mb1-ssd-lite':
        net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
    elif args.net == 'sq-ssd-lite':
        net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
    elif args.net == 'mb2-ssd-lite':
        net = create_mobilenetv2_ssd_lite(len(class_names), width_mult=args.mb2_width_mult, is_test=True)
    else:
        logging.fatal("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        parser.print_help(sys.stderr)
        sys.exit(1)  

    #train_transform = MatchPrior(config.priors, config.center_variance,
    #                              config.size_variance, 0.5)

    #test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)
    # test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)
    # dataset = FolderDataset("/media/wy_disk/wy_file/Detection/dataset/datasets/ECP_Golden_pattern", transform = test_transform)
    # dataset = FolderDataset("/media/wy_disk/wy_file/Detection/dataset/datasets/ECP_Golden_pattern")
    dataset = FolderDataset("/media/wy_disk/ChenYen/VIRAT/dataset_orgnize/val")
    

    timer.start("Load Model")
    net.load(args.trained_model)
    net = net.to(DEVICE)
    print('It took {} seconds to load the model.'.format(timer.end("Load Model")))

    if args.net == 'vgg16-ssd':
        predictor = create_vgg_ssd_predictor(net, nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'mb1-ssd':
        predictor = create_mobilenetv1_ssd_predictor(net, nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'mb1-ssd-lite':
        predictor = create_mobilenetv1_ssd_lite_predictor(net, nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'sq-ssd-lite':
        predictor = create_squeezenet_ssd_lite_predictor(net,nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'mb2-ssd-lite':
        predictor = create_mobilenetv2_ssd_lite_predictor(net, nms_method=args.nms_method, device=DEVICE)
    else:
        logging.fatal("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        parser.print_help(sys.stderr)
        sys.exit(1)
     
    results = []
    eval_path = "eval_results"
    # eval_whole_image(dataset,5, predictor)
    eval_subblock_image(dataset,5, predictor)
    import pdb;pdb.set_trace()
    for i in range(len(dataset)):
        print("process image", i)
        timer.start("Load Image")
        import pdb;pdb.set_trace()
        image = dataset.get_image(i)
        print("Load Image: {:4f} seconds.".format(timer.end("Load Image")))
        timer.start("Predict")
        boxes, labels, probs = predictor.predict(image,10,0.5)
        print("Prediction: {:4f} seconds.".format(timer.end("Predict")))
        # indexes = torch.ones(labels.size(0), 1, dtype=torch.float32) * i
        # print("index:p\t{}".format(sum(probs>0.5)))
        # import pdb;pdb.set_trace()
        boxes,labels,probs = boxes.data.numpy(), labels.data.numpy(), probs.data.numpy()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        for box, _label, _prob in zip(boxes,labels,probs):
            if _prob < 0.7: continue
            print(box)
            box = box.astype(int)
            # import pdb;pdb.set_trace()
            print(box)
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
            # str(str.split(class_names[_label]," ")[1])
            cv2.putText(image, dataset.class_names[_label], (box[0]+20, box[1]+40),cv2.FONT_HERSHEY_SIMPLEX,
                1,  # font scale
                (255, 0, 255),
                2)  # line type
        print(boxes.shape[0])
        cv2.imshow('annotated', image)
        # key = cv2.waitKey(0)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
        
        # cv2.waitKey(300)
def Argparser():
    parser = argparse.ArgumentParser(description="SSD Evaluation on VOC Dataset.")
    parser.add_argument('--net', default="vgg16-ssd",
                        help="The network architecture, it should be of mb1-ssd, mb1-ssd-lite, mb2-ssd-lite or vgg16-ssd.")
    parser.add_argument("--trained_model", type=str)

    parser.add_argument("--dataset_type", default="voc", type=str,
                        help='Specify dataset type. Currently support voc and open_images.')
    parser.add_argument("--dataset", type=str, help="The root directory of the VOC dataset or Open Images dataset.")
    parser.add_argument("--label_file", type=str, help="The label file path.")
    parser.add_argument("--use_cuda", type=str2bool, default=True)
    parser.add_argument("--use_2007_metric", type=str2bool, default=True)
    parser.add_argument("--nms_method", type=str, default="hard")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="The threshold of Intersection over Union.")
    parser.add_argument("--eval_dir", default="eval_results", type=str, help="The directory to store evaluation results.")
    parser.add_argument('--mb2_width_mult', default=1.0, type=float,
                        help='Width Multiplifier for MobilenetV2')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = Argparser()
    main(args)
