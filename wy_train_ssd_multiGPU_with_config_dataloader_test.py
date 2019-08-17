import argparse
import os
import logging
import sys
import itertools

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from detection_configuration import load_model_configuration

from vision.utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from vision.ssd.ssd import MatchPrior
from vision.ssd.vgg_ssd import create_vgg_ssd
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite
from vision.datasets.voc_dataset import VOCDataset
from vision.datasets.open_images import OpenImagesDataset
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.config import vgg_ssd_config
from vision.ssd.config import mobilenetv1_ssd_config
from vision.ssd.config import squeezenet_ssd_config
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform
from vision.datasets.coco_dataset import CocoDetection
from vision.datasets.EuroCity_dataset import EuroCity_Dataset
from vision.datasets.EuroCity_dataset import ECP_subsample_dataset
import cv2
import numpy as np
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def Argments():
    parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
    
    parser.add_argument("--dataset_type", default="voc", type=str,
                    help='Specify dataset type. Currently support voc, coco, open_images, ecp, ecp-random, ecp-centroid.')
    parser.add_argument('--net', default="vgg17-ssd",
                    help="The network architecture, it can be mb2-ssd, mb1-lite-ssd, mb2-ssd-lite or vgg16-ssd.")
    parser.add_argument('--freeze_base_net', action='store_true',
                    help="Freeze base net layers.")
    parser.add_argument('--freeze_net', action='store_true',
                    help="Freeze all the layers except the prediction head.")
    # Params for SGD

    # Params for loading pretrained basenet or checkpoints.
    parser.add_argument('--base_net',
                    help='Pretrained base model')
    parser.add_argument('--pretrained_ssd', help='Pre-trained base model')
    parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
    # Train params
    parser.add_argument('--batch_size', default=33, type=int,
                    help='Batch size for training')
    parser.add_argument('--num_epochs', default=121, type=int,
                    help='the number epochs')
    parser.add_argument('--num_workers', default=16, type=int,
                    help='Number of workers used in dataloading')
    parser.add_argument('--validation_epochs', default=6, type=int,
                    help='the number epochs')
    parser.add_argument('--debug_steps', default=101, type=int,
                    help='Set the debug log output frequency.')
    parser.add_argument('--use_cuda', default=True, type=bool,
                    help='Use CUDA to train model')

    parser.add_argument('--checkpoint_folder', default='models/',
                    help='Directory for saving checkpoint models')
    parser.add_argument('--config', default='config/default_setting.yaml',help = 'configuration')

    args = parser.parse_args()
    
    print(args)
    configuration = load_model_configuration(args.config)
    configuration["flow_control"] = {}
    variable_dict = vars(args)
    for key in variable_dict.keys():
        configuration["flow_control"][key] = variable_dict[key]

    return configuration


def main(args):
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args["flow_control"]["use_cuda"] else "cpu")
    #DEVICE = torch.device("cpu")
    if args["flow_control"]["use_cuda"] and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logging.info("Use Cuda.")

    timer = Timer()

    #logging.info(args)
    if args["flow_control"]["net"] == 'vgg16-ssd':
        create_net = create_vgg_ssd
        config = vgg_ssd_config
    elif args["flow_control"]["net"] == 'mb1-ssd':
        create_net = create_mobilenetv1_ssd
        config = mobilenetv1_ssd_config
    elif args["flow_control"]["net"] == 'mb1-ssd-lite':
        create_net = create_mobilenetv1_ssd_lite
        config = mobilenetv1_ssd_config
    elif args["flow_control"]["net"] == 'sq-ssd-lite':
        create_net = create_squeezenet_ssd_lite
        config = squeezenet_ssd_config
    elif args["flow_control"]["net"] == 'mb2-ssd-lite':
        create_net = lambda num: create_mobilenetv2_ssd_lite(num, width_mult=args['detection_model']["width_mult"])
        config = mobilenetv1_ssd_config
    else:
        logging.fatal("The net type is wrong.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    train_loader, val_loader, num_classes = dataset_loading(args,config)     
    net = create_net(num_classes)
    net, criterion, optimizer, scheduler = optim_and_model_initial(args, net, timer, config, DEVICE)
    dataloader_display(train_loader,net, criterion, optimizer,DEVICE)

def dataloader_display(loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1):
    for i in range(len(loader.dataset)):
        image_id, annotation = loader.dataset.get_annotation(i)
        gt_boxes, classes, is_difficult = annotation
        image = loader.dataset.get_image(i)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        display_flag = False
        show_category = [3]
        for box, label in zip(gt_boxes,classes):
            if label == show_category[0]: #or label== show_category[1]:
                display_flag = True
            else:
                continue
            box = box.astype(int)
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
            #cv2.putText(image, str(label), (box[0]+20, box[1]+40),cv2.FONT_HERSHEY_SIMPLEX,
            cv2.putText(image, str(label), (box[0]+5, box[1]+10),cv2.FONT_HERSHEY_SIMPLEX,
                1,  # font scale
                (255, 0, 255),
                2)  # line type
        if not display_flag: continue
        cv2.imshow('annotated', image)
        if cv2.waitKey(0) & 0xFF==ord('q'):
            break
    
    # for i, data in enumerate(loader):
    #     images, boxes, labels = data
    #     single_image = images[0,...].transpose(0,1).transpose(1,2)
    #     single_image_labels = labels[0,...].data.numpy()
    #     print(single_image_labels.shape)
    #     import pdb;pdb.set_trace()
        
    #     postive_label_x = np.where(single_image_labels!=0)[0]
    #     negtive_label_x = np.where(single_image_labels==0)[0]
    #     single_image_boxes = boxes[0,...]
    #     single_image_pos_boxes = single_image_boxes[postive_label_x]
         
    #     print("Positve Sample / Negtive Sample : {}/{}".format(len(postive_label_x),len(negtive_label_x)))
        
    #     import pdb;pdb.set_trace()
    #     for boxes_index in range(single_image_pos_boxes.shape[0]):
            
    #         box = single_image_pos_boxes[boxes_index]
    #         box[:2] = box[:2] * 300
    #         box[2:] = (2**box[2:]) * 300
    #         label = single_image_labels[postive_label_x[boxes_index]]
    #         print(single_image.max())
    #         print(box)
    #         cv2.rectangle(single_image.data.numpy(), (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)

    #         cv2.putText(single_image, label, (box[0]+20, box[1]+40),cv2.FONT_HERSHEY_SIMPLEX,
    #                 1,  # font scale
    #                 (255, 0, 255),
    #                 2)  # line type
    #         cv2.imshow('annotated', single_image)
        
def dataset_loading(args,config):    

    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)
    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)
    test_normal_transform = TestTransform(config.image_size, 0, 1)

    logging.info("Prepare training datasets.")
    dataset_name = args['flow_control']['dataset_type']
    if dataset_name == 'voc':
        dataset = VOCDataset(dataset_path, transform=train_transform,
                             target_transform=target_transform)
        label_txt_name = "voc-model-labels.txt"
    elif dataset_name == 'open_images':
        dataset = OpenImagesDataset(dataset_path,
             transform=train_transform, target_transform=target_transform,
             dataset_type="train", balance_data=args.balance_data)       
        label_txt_name = "open-images-model-labels.txt"
    elif dataset_name == 'coco':
        dataset = CocoDetection( args["Datasets"]["coco"]["train_image_path"], 
                                 args["Datasets"]["coco"]["train_anno_path"],
                                 transform = test_normal_transform,target_transform=target_transform)
                                #  target_transform=target_transform)
                                #  transform=train_transform, target_transform=target_transform)                              
        label_txt_name = "open-images-model-labels.txt"
    elif dataset_name == "ecp":
        dataset = EuroCity_Dataset( args["Datasets"]["ecp"]["train_image_path"],
                                    args["Datasets"]["ecp"]["train_anno_path"],
                                    transform = test_normal_transform, target_transform = target_transform)
        label_txt_name = "open-images-model-labels.txt"
    elif dataset_name == "ecp-random":
        dataset = ECP_subsample_dataset(args["Datasets"]["ecp"]["train_image_path"],
                                        args["Datasets"]["ecp"]["train_anno_path"],
                                        transform = test_normal_transform, target_transform = target_transform, _sampling_mode = "random", ratio = 0.1)
        label_txt_name = "open-images-model-labels.txt"
    elif dataset_name == "ecp-centroid":
        dataset = ECP_subsample_dataset(args["Datasets"]["ecp"]["train_image_path"],
                                        args["Datasets"]["ecp"]["train_anno_path"],
                                        transform = test_normal_transform, target_transform = target_transform, _sampling_mode = "centroid", ratio = 0.1)
        label_txt_name = "open-images-model-labels.txt"
    else:
        raise ValueError("Dataset type {} is not supported.".format(dataset_name))

    label_file = os.path.join(args["flow_control"]["checkpoint_folder"],label_txt_name)
    if os.path.exists(label_file):
        store_labels(label_file, dataset.class_names)
    logging.info(dataset)
    num_classes = len(dataset.class_names)

    train_dataset = dataset
    logging.info("Stored labels into file {}.".format(label_file))
    logging.info("Train dataset size: {}".format(len(train_dataset)))
    train_loader = DataLoader(train_dataset, args["flow_control"]["batch_size"],
                              num_workers=args["flow_control"]["num_workers"],
                              shuffle=True)
    logging.info("Prepare Validation datasets.")
    if dataset_name == "voc":
        val_dataset = VOCDataset(args.validation_dataset, transform=test_transform,
                                 target_transform=target_transform, is_test=True)
    elif dataset_name == 'open_images':
        val_dataset = OpenImagesDataset(dataset_path,
                                        transform=test_transform, target_transform=target_transform,
                                        dataset_type="test")
        logging.info(val_dataset)
    elif dataset_name == "coco" :
        val_dataset = CocoDetection( args["Datasets"]["coco"]["val_image_path"], 
                                     args["Datasets"]["coco"]["val_anno_path"], 
                                    transform=test_transform, target_transform=target_transform)
        logging.info(val_dataset)
    elif dataset_name == "ecp":
        val_dataset = EuroCity_Dataset( args["Datasets"]["ecp"]["val_image_path"],
                                        args["Datasets"]["ecp"]["val_anno_path"],
                                        transform = test_transform, target_transform = target_transform)
        logging.info(val_dataset)
    elif dataset_name == "ecp-random":
        val_dataset = ECP_subsample_dataset(args["Datasets"]["ecp"]["val_image_path"],
                                            args["Datasets"]["ecp"]["val_anno_path"],
                                            transform = test_transform, target_transform = target_transform, _sampling_mode = "random", ratio = 0.1)
    elif dataset_name == "ecp-centroid":
        val_dataset = ECP_subsample_dataset(args["Datasets"]["ecp"]["val_image_path"],
                                        args["Datasets"]["ecp"]["val_anno_path"],
                                        transform = test_transform, target_transform = target_transform, _sampling_mode = "centroid", ratio = 0.1)

    logging.info("validation dataset size: {}".format(len(val_dataset)))

    val_loader = DataLoader(val_dataset, args["flow_control"]["batch_size"],
                            num_workers=args["flow_control"]["num_workers"],
                            shuffle=False)
    logging.info("Build network.")
    return train_loader, val_loader, num_classes
def optim_and_model_initial(args, net, timer, config, DEVICE):
    #net = create_net(num_classes)
    last_epoch = -1

    base_net_lr = args['Training_hyperparam']['base_net_lr'] if args['Training_hyperparam']['base_net_lr'] !="None" else args['Training_hyperparam']['lr']
    extra_layers_lr = args['Training_hyperparam']['extra_layers_lr'] if args['Training_hyperparam']['extra_layers_lr'] != "None" else args['Training_hyperparam']['lr']
    if args['flow_control']['freeze_base_net']:
        logging.info("Freeze base net.")
        freeze_net_layers(net.base_net)
        params = itertools.chain(net.source_layer_add_ons.parameters(), net.extras.parameters(),net.regression_headers.parameters(), net.classification_headers.parameters())
        params = [
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]
    elif args['flow_control']['freeze_net']:
        freeze_net_layers(net.base_net)
        freeze_net_layers(net.source_layer_add_ons)
        freeze_net_layers(net.extras)
        params = itertools.chain(net.regression_headers.parameters(), net.classification_headers.parameters())
        logging.info("Freeze all the layers except prediction heads.")
    else:
        params = [
            {'params': net.base_net.parameters(), 'lr': base_net_lr},
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]

    timer.start("Load Model")

    if args['flow_control']['resume']:
        logging.info("Resume from the model {}".format(args['flow_control']['resume']))
        net.load(args['flow_control']['resume'])
    elif args['flow_control']['base_net']:
        logging.info("Init from base net {}".format(args['flow_control']['base_net']))
        net.init_from_base_net(args['flow_control']['base_net'])
    elif args['flow_control']['pretrained_ssd']:
        logging.info("Init from pretrained ssd {}".format(args['flow_control']['pretrained_ssd']))
        net.init_from_pretrained_ssd(args['flow_control']['pretrained_ssd'])
    logging.info('Took {:.2f} seconds to load the model.'.format(timer.end("Load Model")))

    # net.to(DEVICE)
    net = nn.DataParallel(net).cuda()
    
    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=DEVICE)
    optimizer = torch.optim.SGD(params, lr=args['Training_hyperparam']['lr'], momentum=args['Training_hyperparam']['momentum'], weight_decay=args['Training_hyperparam']['weighted_decay'])
    logging.info("Learning rate: {}, Base net learning rate: {}, ".format(args['Training_hyperparam']['lr'],base_net_lr)
                 + "Extra Layers learning rate: {}.".format(extra_layers_lr))

    if args['Training_hyperparam']['lr_scheduler'] == 'multi-step':
        logging.info("Uses MultiStepLR scheduler.")
        milestones = [int(v.strip()) for v in args["Training_hyperparam"]["lr_scheduler_param"]["multi-step"]['milestones'].split(",")]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1, last_epoch=last_epoch)
    elif args['Training_hyperparam']['lr_scheduler'] == 'cosine':
        logging.info("Uses CosineAnnealingLR scheduler.")
        scheduler = CosineAnnealingLR(optimizer, float(args['Training_hyperparam']['lr_scheduler_param']['cosine']['t_max']), last_epoch=last_epoch)
    else:
        logging.fatal("Unsupported Scheduler: {}.".format(args['Training_hyperparam']['lr_scheduler']))
        parser.print_help(sys.stderr)
        sys.exit(1)

    logging.info("Start training from epoch {}.".format(last_epoch+1))
    
    return net, criterion, optimizer, scheduler
if __name__=="__main__":
    args =  Argments()
    main(args)
