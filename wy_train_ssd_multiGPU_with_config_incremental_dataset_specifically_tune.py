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
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
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
    parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
    
    parser.add_argument("--dataset_type", default="voc", type=str,
                    help='Specify dataset type. Currently support voc, open_images, ecp, ecp-random and ecp-centroid.')
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

    parser.add_argument('--checkpoint_folder', default='../experiments/models', type = str,
                    help='Directory for saving checkpoint models')
    parser.add_argument('--config', default='config/default_setting.yaml',help = 'configuration')
    parser.add_argument('--dataset_ratio', default = 0.1 , help = "Initial set partial dataset ratio")
    parser.add_argument('--sample_method', type = str, default = 'random', help= "random, sequencial, uncertainty")
    args = parser.parse_args()
    
    print(args)
    configuration = load_model_configuration(args.config)
    configuration["flow_control"] = {}
    variable_dict = vars(args)
    for key in variable_dict.keys():
        configuration["flow_control"][key] = variable_dict[key]

    return configuration

def main_acitve_mode(args):
    # Device setting
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args["flow_control"]["use_cuda"] else "cpu")
    if args["flow_control"]["use_cuda"] and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logging.info("Use Cuda.")

    timer = Timer()

    # Model setting
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
    
    # Dataset 
    train_loader, val_loader, num_classes = dataset_loading(args,config)
    net = create_net(num_classes)
    net, criterion, optimizer, scheduler = optim_and_model_initial(args, net, timer, config, DEVICE)

    Query_iteration = 10
    train_loader.dataset.Active_mode()
    labeled, unlabeled = train_loader.dataset.dataset_information()
    query_item = len(unlabeled) // Query_iteration
    logging.info(
                "Query iteration: {}, ".format(Query_iteration) +
                "per query each item : {} ".format(query_item) 
    )

    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)
    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)
    if args['flow_control']['dataset_type'] == "ecp":
        active_dataset = ECP_table_comm( args["Datasets"]["ecp"]["train_image_path"], 
                                         args["Datasets"]["ecp"]["train_anno_path"], 
                                         transform=test_transform, target_transform=target_transform)
    elif args['flow_control']['dataset_type'] in ["virat", "VIRAT"]:
        active_dataset = ECP_table_comm( args["Datasets"]["virat_seq"]["train_image_path"], 
                                         args["Datasets"]["virat_seq"]["train_anno_path"], 
                                         transform=test_transform, target_transform=target_transform)
    else:
        raise NotImplementedError("Doen't implmented")
    active_dataset.Active_mode()

    for q_iter in range(Query_iteration):
        if q_iter != 0 :
            scheduler.base_lrs[0] = scheduler.base_lrs[0] * 1.1
        active_dataset.Active_mode()
        labeled, unlabeled = active_dataset.dataset_information()
        logging.info(
            "Query iteration: {}/{}, ".format(q_iter,Query_iteration) +
            "per query each item : {} ".format(query_item) 
        )
        logging.info("Fetch data...")
        # imgs_list, bboxes_list, labels_list = train_loader.dataset.data_fetch()
        logging.info("Fetch data finish...")
        _setting_sampler = args["flow_control"]["sample_method"]
        if _setting_sampler == "random":
            net.train(False)
            query_index = np.random.choice(unlabeled,query_item,replace=False)
            train_loader.dataset.setting_be_selected_sample(query_index)
            active_dataset.setting_be_selected_sample(query_index)
        elif _setting_sampler == "seqencial":            
            net.train(False)
            query_index = unlabeled[:query_item]
            train_loader.dataset.setting_be_selected_sample(query_index)
            active_dataset.setting_be_selected_sample(query_index)
        elif _setting_sampler == "uncertainty_modify":
            net.train(False)
            imgs_list = active_dataset.data_fetch()
            max_num = 50
            confidences = []
            for index in range( len(imgs_list)//max_num + 1 ):
                with torch.no_grad():
                    begin_pointer = index * max_num
                    end_pointer = min( (index+1) * max_num, len(imgs_list))
                    sub_batch = torch.stack(imgs_list[begin_pointer:end_pointer]).cuda()
                    _confidence, locations = net(sub_batch)
                    confidences.append(_confidence.data.cpu())
            confidences = torch.cat(confidences, 0)
            probability = torch.softmax(confidences, 2)
            entropy = torch.sum(probability * torch.log(probability) * -1, 2)
            mean = torch.mean(entropy, 1)
            stddev = torch.std(entropy, 1)
            criteria = mean * stddev / (mean+stddev)
            query_index = torch.argsort( -1 * criteria)[:query_item].tolist()
            train_loader.dataset.setting_be_selected_sample(query_index)
            active_dataset.setting_be_selected_sample(query_index)
        elif _setting_sampler == "uncertainty":
            net.train(False)
            imgs_list = active_dataset.data_fetch()
            max_num = 50
            confidences = []
            for index in range( len(imgs_list)//max_num + 1 ):
                with torch.no_grad():
                    begin_pointer = index * max_num
                    end_pointer = min( (index+1) * max_num, len(imgs_list))
                    sub_batch = torch.stack(imgs_list[begin_pointer:end_pointer]).cuda()
                    _confidence, locations = net(sub_batch)
                    confidences.append(_confidence.data.cpu())
            confidences = torch.cat(confidences, 0)
            probability = torch.softmax(confidences, 2)
            entropy = torch.sum(probability * torch.log(probability) * -1, 2)
            maximum = torch.max(entropy, 1)[0]
            criteria = maximum
            query_index = torch.argsort( -1 * criteria)[:query_item].tolist()
            train_loader.dataset.setting_be_selected_sample(query_index)
            active_dataset.setting_be_selected_sample(query_index)
        elif _setting_sampler == "diversity":
            pass
        elif _setting_sampler == "balance_feature":
            pass
        else:
            raise NotImplementedError("_setting_sampler : {} doesn't implement".format(_setting_sampler))
        

        train_loader.dataset.training_mode()
        # Training process
        for epoch in range(0, args['flow_control']['num_epochs']):

            scheduler.step()
            train(args, train_loader, net, criterion, optimizer, device=DEVICE, debug_steps=args['flow_control']['debug_steps'], epoch=epoch)
            
            if epoch % args['flow_control']['validation_epochs'] == 0 or epoch == args['flow_control']['num_epochs'] - 1:
                val_loss, val_regression_loss, val_classification_loss = test(args, val_loader, net, criterion, DEVICE)
                logging.info(
                    "Epoch: {}, ".format(epoch) +
                    "Validation Loss: {:.4f}, ".format(val_loss) +
                    "Validation Regression Loss {:.4f}, ".format(val_regression_loss) +
                    "Validation Classification Loss: {:.4f}".format(val_classification_loss)
                )
                
                _postfix_infos = [args['flow_control']['dataset_type'], args["flow_control"]["net"],START_TRAINING_TIME] if (args['flow_control']['dataset_type'] != "ecp-random" or args['flow_control']['dataset_type'] != "ecp-centroid") \
                                else [args['flow_control']['dataset_type'], str(args['flow_control']['dataset_ratio']), args["flow_control"]["net"],START_TRAINING_TIME] 
                postfix = "_".join(_postfix_infos)
                folder_name = os.path.join(args["flow_control"]["checkpoint_folder"] + "_" + postfix,"query_iter_{}".format(str(float(q_iter+1)/float(Query_iteration))))
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)
                model_path = os.path.join(folder_name, "{}-Epoch-{}-Loss-{}.pth".format(args['flow_control']['net'],epoch,val_loss))
                net.module.save(model_path)
                logging.info("Saved model {}".format(model_path))

def main(args):
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args["flow_control"]["use_cuda"] else "cpu")
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
    
    for epoch in range(0, args['flow_control']['num_epochs']):
        scheduler.step()
        train(args, train_loader, net, criterion, optimizer, device=DEVICE, debug_steps=args['flow_control']['debug_steps'], epoch=epoch)
        
        if epoch % args['flow_control']['validation_epochs'] == 0 or epoch == args['flow_control']['num_epochs'] - 1:
            val_loss, val_regression_loss, val_classification_loss = test(args, val_loader, net, criterion, DEVICE)
            logging.info(
                "Epoch: {}, ".format(epoch) +
                "Validation Loss: {:.4f}, ".format(val_loss) +
                "Validation Regression Loss {:.4f}, ".format(val_regression_loss) +
                "Validation Classification Loss: {:.4f}".format(val_classification_loss)
            )
            
            _postfix_infos = [args['flow_control']['dataset_type'], args["flow_control"]["net"],START_TRAINING_TIME] if (args['flow_control']['dataset_type'] != "ecp-random" or args['flow_control']['dataset_type'] != "ecp-centroid") \
                            else [args['flow_control']['dataset_type'], str(args['flow_control']['dataset_ratio']), args["flow_control"]["net"],START_TRAINING_TIME] 
            postfix = "_".join(_postfix_infos)
            folder_name = args["flow_control"]["checkpoint_folder"] + "_" + postfix
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            model_path = os.path.join("{}-Epoch-{}-Loss-{}.pth".format(args['flow_control']['net'],epoch,val_loss))
            net.module.save(model_path)
            logging.info("Saved model {}".format(model_path))

def train(args, loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1):
    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    dataset_name = args['flow_control']['dataset_type']
    weighted_vector = None
    if dataset_name in ['ecp','ecp-random','ecp-centroid']:
        weighted_vector = torch.zeros(13)
        weighted_vector[0:5] = 1
        weighted_vector =weighted_vector.cuda()
    elif dataset_name in ["VIRAT", "virat"]:        
        weighted_vector = torch.zeros(7)
        weighted_vector[0] = 1
        weighted_vector[2:5] = 1
        weighted_vector[6] = 1
        weighted_vector = weighted_vector.cuda()
    legal_label = {i:0 for i in range(7)}
    for i, data in enumerate(loader):
        images, boxes, labels = data
        # sta_map = (torch.isinf(boxes[...,2]) + torch.isinf(boxes[...,3])) ==0
        # item = torch.sum(sta_map)
        # item_label = labels[sta_map]
        
        # for index in range(7):
        #     legal_label[index] += torch.sum(item_label==index).item()
        #import pdb;pdb.set_trace()
        # continue
        
        labels = labels.type("torch.LongTensor")
        
        logging.debug("==== Enum img shape {} & type {} =====".format(images.size(), images.type()))
        logging.debug("==== Enum boxes shape {} & type {} =====".format(boxes.size(), boxes.type()))
        logging.debug("==== Enum label shape {} & type {} =====".format(labels.size(), labels.type()))

        logging.debug("------ Enumerate : {} -------".format(i))
        
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        logging.debug("Pass throgh")
        optimizer.zero_grad()
        confidence, locations = net(images)
        #import pdb;pdb.set_trace() 
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes,weighted_vector)  # TODO CHANGE BOXES
        loss = regression_loss + classification_loss #* 0.3
        if torch.isnan(loss):
            continue
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        if i and i % debug_steps == 0:
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps
            logging.info(
                "Epoch: {}, Step: {}, ".format(epoch,i) +
                "Average Loss: {:.4f}, ".format(avg_loss) +
                "Average Regression Loss {:.4f}, ".format(avg_reg_loss) +
                "Average Classification Loss: {:.4f}".format(avg_clf_loss)
            )
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0
    # print(legal_label)
    # import pdb;pdb.set_trace()


def test(args,loader, net, criterion, device):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    dataset_name = args['flow_control']['dataset_type']
    weighted_vector = None
    if dataset_name == 'ecp':
        weighted_vector = torch.zeros(13)
        weighted_vector[0:5] = 1
        weighted_vector =weighted_vector.cuda()
    elif dataset_name in ["VIRAT", "virat"]:        
        weighted_vector = torch.zeros(7)
        weighted_vector[0] = 1
        weighted_vector[2:5] = 1
        weighted_vector[6] = 1
        weighted_vector =weighted_vector.cuda()
    for _, data in enumerate(loader):
        images, boxes, labels = data
        # import pdb;pdb.set_trace()
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes,weighted_vector)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
    return running_loss / num, running_regression_loss / num, running_classification_loss / num
def dataset_loading(args,config):    

    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)
    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

    logging.info("Prepare training datasets.")
    dataset_name = args['flow_control']['dataset_type']
    if dataset_name == 'voc':
        dataset = VOCDataset("", transform=train_transform,
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
                                 transform=train_transform, target_transform=target_transform)
        label_txt_name = "open-images-model-labels.txt"
    elif dataset_name == 'ecp':
        #dataset = EuroCity_Dataset( args["Datasets"]["ecp"]["train_image_path"], 
        #                            args["Datasets"]["ecp"]["train_anno_path"], 
        #                            transform=train_transform, target_transform=target_transform)        
        dataset = ECP_table_comm( args["Datasets"]["ecp"]["train_image_path"], 
                                   args["Datasets"]["ecp"]["train_anno_path"], 
                                   transform=train_transform, target_transform=target_transform)
        dataset.Active_mode()
        if len(dataset) == 0:
            raise ValueError("Doesn't exist any file")

        label_txt_name = "open-images-model-labels.txt"
    elif dataset_name == "ecp-random":
        dataset = ECP_subsample_dataset( args["Datasets"]["ecp"]["train_image_path"], 
                                         args["Datasets"]["ecp"]["train_anno_path"], 
                                         transform=train_transform, target_transform=target_transform, _sampling_mode = "random", ratio = args['flow_control']['dataset_ratio'])
        label_txt_name = "open-images-model-labels.txt"
    elif dataset_name == "ecp-centroid":
        dataset = ECP_subsample_dataset( args["Datasets"]["ecp"]["train_image_path"], 
                                         args["Datasets"]["ecp"]["train_anno_path"], 
                                         transform=train_transform, target_transform=target_transform, _sampling_mode = "centroid", ratio = args['flow_control']['dataset_ratio'])
        label_txt_name = "open-images-model-labels.txt"
    elif dataset_name in ["virat","VIRAT"]:
        # dataset = VIRAT_Loader(args["Datasets"]["virat"]["train_image_path"], 
        #                        args["Datasets"]["virat"]["train_anno_path"],
        #                        transform=train_transform, target_transform=target_transform)
        dataset = VIRAT_Dataset(args["Datasets"]["virat_seq"]["train_image_path"], 
                               args["Datasets"]["virat_seq"]["train_anno_path"],
                               transform=train_transform, target_transform=target_transform, downpurning_ratio = 0.2) #0.2
        # dataset = VIRAT_table_comm(args["Datasets"]["virat_seq"]["train_image_path"], 
        #                        args["Datasets"]["virat_seq"]["train_anno_path"],
        #                        transform=train_transform, target_transform=target_transform)
        
        
        label_txt_name = "virat_labels.txt"
    else:
        raise ValueError("Dataset type {} is not supported.".format(dataset_name))

    label_file = os.path.join(args["flow_control"]["checkpoint_folder"], label_txt_name)
    if os.path.exists(label_file):
        store_labels(label_file, dataset.class_names)
    logging.info(dataset)
    num_classes = len(dataset.class_names)
    
    train_dataset = dataset
    logging.info("Stored labels into file {}.".format(label_file))
    logging.info("Train dataset size: {}".format(len(train_dataset)))
    logging.debug("================= train_loader ===================")
    logging.debug("DataLoader batchsize : ",args["flow_control"]["batch_size"])
    # if dataset_name == "virat":
    #     indicies = np.arange(args["flow_control"]["batch_size"])
    #     train_loader = DataLoader(train_dataset, args["flow_control"]["batch_size"],
    #                           num_workers=args["flow_control"]["num_workers"],
    #                           shuffle=False, sampler=SubsetRandomSampler(indicies))
    # else:
    train_loader = DataLoader(train_dataset, args["flow_control"]["batch_size"],
                            num_workers=args["flow_control"]["num_workers"],
                            shuffle=True)
    logging.info("Prepare Validation datasets.")
    if dataset_name == "voc":
        raise NotImplementedError("Doesn't modify")
        val_dataset = VOCDataset("", transform=test_transform,
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
    elif dataset_name in ["ecp","ecp-random","ecp-centroid"] :
        val_dataset = EuroCity_Dataset( args["Datasets"]["ecp"]["val_image_path"], 
                                        args["Datasets"]["ecp"]["val_anno_path"], 
                                        transform=test_transform, target_transform=target_transform)
    # elif dataset_name = "ecp-random":
    #     val_dataset = ECP_subsample_dataset( args["Datasets"]["ecp"]["val_image_path"], 
    #                                          args["Datasets"]["ecp"]["val_anno_path"], 
    #                                          transform=test_transform, target_transform=target_transform, _sampling_mode = "random", ratio = 0.1)
    # elif dataset_name = "ecp-centroid":
    #     val_dataset = ECP_subsample_dataset( args["Datasets"]["ecp"]["val_image_path"], 
    #                                          args["Datasets"]["ecp"]["val_anno_path"], 
    #                                          transform=test_transform, target_transform=target_transform, _sampling_mode = "centroid", ratio = 0.1)
    
    elif dataset_name in ["virat", "VIRAT"]:
        # val_dataset = VIRAT_Loader(args["Datasets"]["virat"]["test_image_path"], 
        #                            args["Datasets"]["virat"]["test_anno_path"],
        #                            transform=train_transform, target_transform=target_transform)
        val_dataset = VIRAT_Dataset(args["Datasets"]["virat_seq"]["train_image_path"], 
                                    args["Datasets"]["virat_seq"]["train_anno_path"],
                                    transform=train_transform, target_transform=target_transform,downpurning_ratio = 0.2 * 3./9.)


    logging.info("validation dataset size: {}".format(len(val_dataset)))
    # if dataset_name == "virat":
    #     indicies = np.arange(args["flow_control"]["batch_size"])
    #     val_loader = DataLoader(train_dataset, args["flow_control"]["batch_size"],
    #                           num_workers=args["flow_control"]["num_workers"],
    #                           shuffle=False, sampler=SubsetRandomSampler(indicies))
    # else:
    #     val_loader = DataLoader(val_dataset, args["flow_control"]["batch_size"],
    #                             num_workers=args["flow_control"]["num_workers"],
    #                             shuffle=False)
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
    neg_pos_ratio = 3 #3
    
    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=neg_pos_ratio,
                             center_variance=0.1, size_variance=0.2, device=DEVICE)
    # criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=1,
    #                          center_variance=0.1, size_variance=0.2, device=DEVICE)
    optimizer = torch.optim.SGD(params, lr=args['Training_hyperparam']['lr'], momentum=args['Training_hyperparam']['momentum'], weight_decay=args['Training_hyperparam']['weighted_decay'])
    logging.info("Learning rate: {}, Base net learning rate: {}, ".format(args['Training_hyperparam']['lr'],base_net_lr)
                 + "Extra Layers learning rate: {}.".format(extra_layers_lr))

    if args['Training_hyperparam']['lr_scheduler'] == 'multi-step':
        logging.info("Uses MultiStepLR scheduler.")
        milestones = [int(v.strip()) for v in args["Training_hyperparam"]["lr_scheduler_param"]["multi-step"]['milestones'].split(",")]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma = args["Training_hyperparam"]["lr_scheduler_param"]["multi-step"]['gamma'], last_epoch=last_epoch)
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
    # main_acitve_mode(args)
