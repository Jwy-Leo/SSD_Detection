import argparse
import os
import logging
import sys
import itertools

import torch
import numpy as np
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
from vision.datasets.EuroCity_dataset_incremental import ECP_table_comm
# from vision.datasets.VIRAT_DataLoader import VIRAT_Loader
from vision.datasets.VIRAT_DataLoader_V2 import VIRAT_Dataset, VIRAT_table_comm
from active_module_sampler.active_learning_sampler import random_sampler, uncertainty_sampler
from torch.utils.data.sampler import SubsetRandomSampler
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
import time 
# import pynvml
# pynvml.nvmlInit()
# handle = pynvml.nvmlDeviceGetHandleByIndex(0)
def memory_info():
    float32_bytes = 4
    free_Mem_bytes = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
    # free_Mem_bytes = pynvml.nvmlDeviceGetMemoryInfo(handle).free
    float32_item = free_Mem_bytes / float32_bytes
    return float32_item
START_TRAINING_TIME = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) 
def Argments():
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With Pytorch')
    
    # Train params
    parser.add_argument('--batch_size', default=33, type=int,
                    help='Batch size for training')
    parser.add_argument('--num_epochs', default=121, type=int,
                    help='the number epochs')
    parser.add_argument('--num_workers', default=16, type=int,
                    help='Number of workers used in dataloading')
    parser.add_argument('--config', default='config/prague_combine_balance.yaml',help = 'configuration')
    args = parser.parse_args()
    
    print(args)
    configuration = load_model_configuration(args.config)
    configuration["flow_control"] = {}
    variable_dict = vars(args)
    for key in variable_dict.keys():
        configuration["flow_control"][key] = variable_dict[key]

    return configuration
def main_active_mode(args):
    Query_iteration = 10
    create_net = lambda num: create_mobilenetv2_ssd_lite(num, width_mult=args['detection_model']["width_mult"])
    config = mobilenetv1_ssd_config
    train_loader, val_loader, num_classes = dataset_loading(args,config)

    target_transform = MatchPrior(config.priors, config.center_variance, config.size_variance, 0.5)
    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)
    active_dataset = VIRAT_table_comm( args["Datasets"]["virat_seq"]["train_image_path"],
                                       args["Datasets"]["virat_seq"]["train_anno_path"],
                                       transform = test_transform, target_transform = target_transform, downpurning_ratio = 0.2)

    labeled, unlabeled = train_loader.dataset.dataset_information()
    query_item = len(unlabeled) // Query_iteration

    active_dataset.Active_mode()

    for q_iter in range(Query_iteration):
        #if q_iter != 0:
        active_dataset.Active_mode()
        labeled, unlabeled = active_dataset.dataset_information()
        query_index = np.random.choice(unlabeled,query_item, replace = False)
        train_loader.dataset.setting_be_selected_sample(query_index)
        active_dataset.setting_be_selected_sample(query_index)
        train_loader.dataset.training_mode()
        train(args, train_loader)
def main(args):
    
    create_net = lambda num: create_mobilenetv2_ssd_lite(num, width_mult=args['detection_model']["width_mult"])
    config = mobilenetv1_ssd_config
    

    train_loader, val_loader, num_classes = dataset_loading(args,config)
    
    
    for epoch in range(0, args['flow_control']['num_epochs']):
        train(args, train_loader)
        
        # if epoch % args['flow_control']['validation_epochs'] == 0 or epoch == args['flow_control']['num_epochs'] - 1:
        #     val_loss, val_regression_loss, val_classification_loss = test(args, val_loader)
        #     logging.info(
        #         "Epoch: {}, ".format(epoch) +
        #         "Validation Loss: {:.4f}, ".format(val_loss) +
        #         "Validation Regression Loss {:.4f}, ".format(val_regression_loss) +
        #         "Validation Classification Loss: {:.4f}".format(val_classification_loss)
        #     )

def train(args, loader):
    
    legal_label = {i:0 for i in range(7)}
    for i, data in enumerate(loader):
        images, boxes, labels = data
        sta_map = (torch.isinf(boxes[...,2]) + torch.isinf(boxes[...,3])) ==0
        item = torch.sum(sta_map)
        item_label = labels[sta_map]
        
        for index in range(7):
            legal_label[index] += torch.sum(item_label==index).item()
        
        labels = labels.type("torch.LongTensor")
        
        logging.debug("==== Enum img shape {} & type {} =====".format(images.size(), images.type()))
        logging.debug("==== Enum boxes shape {} & type {} =====".format(boxes.size(), boxes.type()))
        logging.debug("==== Enum label shape {} & type {} =====".format(labels.size(), labels.type()))

        logging.debug("------ Enumerate : {} -------".format(i))
        continue
    print(legal_label)
    import pdb;pdb.set_trace()


def test(args,loader):
    for _, data in enumerate(loader):
        images, boxes, labels = data
        # import pdb;pdb.set_trace()
        
def dataset_loading(args,config):    

    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)
    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

    #dataset = VIRAT_Dataset(args["Datasets"]["virat_seq"]["train_image_path"], 
    #                        args["Datasets"]["virat_seq"]["train_anno_path"],
    #                        transform=train_transform, target_transform=target_transform, downpurning_ratio = 0.2) #0.2
    dataset = VIRAT_table_comm( args["Datasets"]["virat_seq"]["train_image_path"],
                                args["Datasets"]["virat_seq"]["train_anno_path"],
                                transform = train_transform, target_transform = target_transform, downpurning_ratio = 0.2)
    # dataset = VIRAT_Dataset(args["Datasets"]["virat_seq"]["train_image_path"], 
    #                         args["Datasets"]["virat_seq"]["train_anno_path"],
    #                         transform=train_transform, downpurning_ratio = 0.05) #0.2
     
    label_file = ""
    if os.path.exists(label_file):
        store_labels(label_file, dataset.class_names)
    logging.info(dataset)
    num_classes = len(dataset.class_names)
    
    train_dataset = dataset
    
    train_loader = DataLoader(train_dataset, args["flow_control"]["batch_size"],
                            num_workers=args["flow_control"]["num_workers"],
                            shuffle=True)
    
    val_dataset = VIRAT_Dataset(args["Datasets"]["virat_seq"]["train_image_path"], 
                                args["Datasets"]["virat_seq"]["train_anno_path"],
                                transform=train_transform, target_transform=target_transform,downpurning_ratio = 0.2 * 3./9.)
    val_loader = DataLoader(val_dataset, args["flow_control"]["batch_size"],
                            num_workers=args["flow_control"]["num_workers"],
                            shuffle=False)
    logging.info("Build network.")
    return train_loader, val_loader, num_classes
if __name__=="__main__":
    args =  Argments()
    #main(args)
    main_active_mode(args)
    # main_acitve_mode(args)
