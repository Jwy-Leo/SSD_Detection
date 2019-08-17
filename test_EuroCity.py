import os
import json
import re
import zipfile
import numpy as np
import torch
import torch.utils.data as data
import cv2
from torch.utils.data import DataLoader
from vision.ssd.data_preprocessing import TrainAugmentation
from vision.ssd.config import mobilenetv1_ssd_config
from vision.ssd.ssd import MatchPrior
# def load_data_ecp(goundtruth_path, det_path, gt_ext='.json',det_ext=".json"):
from vision.datasets.EuroCity_dataset import EuroCity_Dataset
def try_the_datasets():
    # Transform
    config = mobilenetv1_ssd_config
    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance, config.size_variance, 0.5)
    # train test
    # img_path = "/media/wy_disk/wy_file/Detection/dataset/datasets/ECP/ECP/day/img/train"
    # label_path = "/media/wy_disk/wy_file/Detection/dataset/datasets/ECP/ECP/day/labels/train"

    # val test
    # img_path = "/media/wy_disk/wy_file/Detection/dataset/datasets/ECP/ECP/day/img/val"
    # label_path = "/media/wy_disk/wy_file/Detection/dataset/datasets/ECP/ECP/day/labels/val"

    # night train test
    # img_path = "/media/wy_disk/wy_file/Detection/dataset/datasets/ECP/ECP/night/img/train"
    # label_path = "/media/wy_disk/wy_file/Detection/dataset/datasets/ECP/ECP/night/labels/train"

    # night val test
    # img_path = "/media/wy_disk/wy_file/Detection/dataset/datasets/ECP/ECP/night/img/val"
    # label_path = "/media/wy_disk/wy_file/Detection/dataset/datasets/ECP/ECP/night/labels/val"

    dataset = EuroCity_Dataset(img_path, label_path, transform = train_transform, target_transform = target_transform)
    DL = DataLoader(dataset,batch_size=3,shuffle=False,num_workers=0)
    # for index, (img, bbox, labels) in enumerate(DL):
    for index, D in enumerate(DL):
        import pdb;pdb.set_trace()
try_the_datasets()
