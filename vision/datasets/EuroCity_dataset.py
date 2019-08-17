import os
import json
import re
import zipfile
import numpy as np
import torch
import torch.utils.data as data
import cv2
#import sys
#sys.path.pop(0)
#sys.path.append("/media/wy_disk/wy_file/Detection/detection_models/pytorch-train_ECP/")
#print(sys.path)
from torch.utils.data import DataLoader
#from vision.ssd.data_preprocessing import TrainAugmentation
#from vision.ssd.ssd import MatchPrior
#from vision.ssd.config import mobilenetv1_ssd_config
# def load_data_ecp(goundtruth_path, det_path, gt_ext='.json',det_ext=".json"):
instance_tag = [u'occluded>10',u'occluded>40',u'occluded>80',u'unsure_orientation',u'truncated>10',u'depiction',u'truncated>40',u'truncated>80',u'sitting-lying',u'behind-glass',u'skating']
def try_the_datasets():
    # Transform
    #config = mobilenetv1_ssd_config
    #train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    #target_transform = MatchPrior(config.priors, config.center_variance, config.size_variance, 0.5)
    train_transform = None
    target_transform = None
    # dataset = EuroCity_Dataset("/media/wy_disk/wy_file/Detection/dataset/datasets/ECP/ECP/day/img/train","/media/wy_disk/wy_file/Detection/dataset/datasets/ECP/ECP/day/labels/train")
    dataset = EuroCity_Dataset("/media/wy_disk/wy_file/Detection/dataset/datasets/ECP/other_temp/ECP/day/img/val","/media/wy_disk/wy_file/Detection/dataset/datasets/ECP/ECP/day/labels/val", transform = train_transform, target_trainsform = target_transform)
    # dataset = EuroCity_Dataset("/media/wy_disk/wy_file/Detection/dataset/datasets/ECP/ECP/night/img/val","/media/wy_disk/wy_file/Detection/dataset/datasets/ECP/ECP/night/labels/val")
    #dataset = EuroCity_Dataset("/media/wy_disk/wy_file/Detection/dataset/datasets/ECP/ECP/night/img/train","/media/wy_disk/wy_file/Detection/dataset/datasets/ECP/ECP/night/labels/train")
    DL = DataLoader(dataset,batch_size=3,shuffle=False,num_workers=0)
    # for index, (img, bbox, labels) in enumerate(DL):
    for index, D in enumerate(DL):
        import pdb;pdb.set_trace()
class EuroCity_Dataset(data.Dataset):
    def __init__(self, dataset_city_image_folder_path, dataset_city_GT_path, label_file = "xxx.txt",
                       transform = None, target_transform = None, transforms = None):
        super(EuroCity_Dataset,self).__init__()
        self._check_path_exists(dataset_city_image_folder_path,dataset_city_GT_path)
        if os.path.exists(label_file):
            with open(label_file,"r") as F:
                lines = F.readlines()
            self.class_names = [line[:-1] for line in lines]
            # self.class_names.insert(0,"BackGround")
        else:
            self._class_names = { u"BackGround":0,u'pedestrian':1,u'rider':2,u'bicycle':3,u'motorbike':4,u'person-group-far-away':5,u'motorbike-group':6,u'scooter-group':7,u'buggy-group':8,u'bicycle-group':9,u"rider+vehicle-group-far-away":10,u'wheelchair-group':11,u'tricycle-group':12}
            # self._class_names = {u"BackGround":0,u'person-group-far-away':1,u'motorbike-group':2,u'pedestrian':3,u'scooter-group':4,u'buggy-group':5,u'bicycle-group':6,u'rider':7,u"rider+vehicle-group-far-away":8,u'bicycle':9,u'wheelchair-group':10,u'tricycle-group':11,u'motorbike':12}
            # xxx group it mean that ignore region
            self.class_names = ["BackGround",'pedestrian','rider','bicycle','motorbike','DK','DK','DK','DK',"DK","DK","DK","DK"]
            # bicycle dirty, it circle multiple bicycle
        self.image_path = []
        self.gt_path = []
        self._city_number_imgs = {}
        self._city_cumsum = {}
        
        _mpth, _mfolder, _mfile = next(iter(os.walk(dataset_city_image_folder_path)))
        self._city_name = _mfolder
        
        cumsum = 0
        for _city_name in _mfolder:

            # Loading groud truth path
            sub_path = os.path.join(dataset_city_GT_path,_city_name)
            _, _, _gt_paths = next(iter(os.walk(sub_path)))
            _gt_paths = sorted(_gt_paths)
            remove_indexs = self._non_boxes_faliure_case(sub_path,_gt_paths)

            remove_item = [_gt_paths[_i] for _i in remove_indexs]
            for _item in remove_item : _gt_paths.remove(_item)

            _number_of_datapair = len(_gt_paths)

            _gt_paths = [out for out in map(os.path.join,[sub_path for i in range(_number_of_datapair)],_gt_paths) ]
            self.gt_path.extend(_gt_paths)

            # Loading image path
            sub_path = os.path.join(_mpth,_city_name)
            _, _, _img_paths = next(iter(os.walk(sub_path)))
            _img_paths = sorted(_img_paths)
            
            remove_item = [_img_paths[_i] for _i in remove_indexs]
            for _item in remove_item : _img_paths.remove(_item)

            _img_paths = [out for out in map(os.path.join,[sub_path for i in range(_number_of_datapair)],_img_paths) ]
            self.image_path.extend(_img_paths)
                        
            # Meta information
            self._city_number_imgs[_city_name] = _number_of_datapair
            self._city_cumsum[_city_name] = cumsum
            cumsum += _number_of_datapair
        self.transform = transform 
        self.transforms = transforms
        self.target_transform = target_transform
    def __getitem__(self,index):
        #img_path = self.image_path[index]
        #gt_path = self.gt_path[index]
        #img = self._load_image(img_path)
        #bboxes_x0y0x1y1, labels = self._load_gt(gt_path)
        img = self.get_image(index)        
        index, (bboxes_x0y0x1y1, labels, difficult) = self.get_annotation(index)
        if self.transform:
            img, bboxes_x0y0x1y1, labels = self.transform(img,bboxes_x0y0x1y1,labels)
        if self.target_transform:
            bboxes_x0y0x1y1, labels = self.target_transform(bboxes_x0y0x1y1,labels)
        return img, bboxes_x0y0x1y1, labels
    def __len__(self):
        return len(self.image_path)
    
    def _check_path_exists(self,dataset_city_image_folder_path,dataset_city_GT_path):
        assert os.path.exists(dataset_city_image_folder_path), 'The folder path : {} is wrong '.format(dataset_city_image_folder_path)
        assert os.path.exists(dataset_city_GT_path), 'The gound truth path : {} is wrong '.format(dataset_city_GT_path)
    def get_image(self,index):
        return self._load_image(self.image_path[index])
    def get_annotation(self,index):
        bboxes, classes = self._load_gt(self.gt_path[index])
        difficult = np.zeros_like(classes)
        return index, (bboxes, classes, difficult)
    def _load_image(self,img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    def _load_gt(self,gt_path):
        with open(gt_path,"r") as F:
            data = json.load(F)
        #print(data)
        data = data[u'children']
        bboxes, labels = [], []
        for _item in data:
            tag_flag = False
            for sub_tag in _item[u'tags']:
                tag_flag = tag_flag | self._filtering_unresearable_case(sub_tag)                
                #if sub_tag not in tag:
                #    raise ValueError, "{} not in tag list".format(sub_tag)
            if tag_flag :
                continue
            bboxes.append(np.array([_item[u'x0'],_item[u'y0'],_item[u'x1'],_item[u'y1']]))
            labels.append(self._class_names[_item[u'identity']])
        bboxes = np.vstack(bboxes).astype(np.float32)
        labels = np.array(labels).astype(np.int64)
        return bboxes, labels
    def _non_boxes_faliure_case(self,sub_path,gt_paths):
        
        _gt_paths = [out for out in map(os.path.join,[sub_path for i in range(len(gt_paths))],gt_paths) ]
        indexs = []
        for index, path in enumerate(_gt_paths):
            with open(path,"r") as F:
                data = json.load(F)
            _num_boxes_case = len(data[u'children'])
            _num_filter = 0
            for _item in data[u'children']:
                tag_flag = False
                for sub_tag in _item[u'tags']:                    
                    tag_flag = tag_flag | self._filtering_unresearable_case(sub_tag)
                if tag_flag:
                    _num_filter +=1
            if (_num_boxes_case - _num_filter) ==0:
                indexs.append(index)
        return indexs
    def _load_label_name_file(self):
        pass
    def _filtering_unresearable_case(self,_instance_tag):
        _filter_tag = [u'occluded>80',u'truncated>80',u'depicition']#,u'unsure_orientation']#,u'occluded>40',u'truncated>40']
        # instance_tag = [u'occluded>10',u'occluded>40',u'occluded>80',u'unsure_orientation',u'truncated>10',u'depiction',u'truncated>40',u'truncated>80',u'sitting-lying',u'behind-glass',u'skating']
        if _instance_tag in _filter_tag:
            return True
        else:
            return False
class ECP_subsample_dataset(EuroCity_Dataset):
    def __init__(self, dataset_city_image_folder_path, dataset_city_GT_path, label_file = "xxx.txt",
                       transform = None, target_transform = None, transforms = None,_sampling_mode=None, ratio = 0.1):
        input_args = ( dataset_city_image_folder_path, dataset_city_GT_path, label_file,
                    transform, target_transform, transforms)
        super(ECP_subsample_dataset,self).__init__(*input_args)
        if _sampling_mode is None:
            raise ValueError("Doesn't define sampling mode.")
        else:
            sampler = self._random_sample if _sampling_mode=="random" else self._centroid_sample
        sampler(ratio)
    def _random_sample(self,ratio):
        cumsum, city_list = self._get_cumsum_list_and_cityname()
        new_image_path, new_gt_path = [], []
        new_city_nubmer_imgs, new_city_cumsum = {}, {}

        accumsum = 0
        for _index, begin_index in enumerate(cumsum):
            name_key = city_list[_index]
            num_imgs = self._city_number_imgs[name_key]
            _sample_number = num_imgs // int(1.0/ratio)
            
            # sampler
            _sampled_index = np.random.choice([i+begin_index for i in range(num_imgs)],_sample_number)
            for SI in _sampled_index:
                new_image_path.append(self.image_path[SI])
                new_gt_path.append(self.gt_path[SI])
         
            new_city_nubmer_imgs[name_key] = num_imgs
            new_city_cumsum[name_key] = accumsum
            accumsum += num_imgs

        self.image_path = new_image_path
        self.gt_path = new_gt_path
        self._city_number_imgs = new_city_nubmer_imgs
        self._city_cumsum = new_city_cumsum
    def _centroid_sample(self,ratio):
        cumsum, city_list = self._get_cumsum_list_and_cityname()
        new_image_path, new_gt_path = [], []
        new_city_nubmer_imgs, new_city_cumsum = {}, {}
        accumsum = 0
        for _index, begin_index in enumerate(cumsum):
            name_key = city_list[_index]
            num_imgs = self._city_number_imgs[name_key]
            _sample_number = num_imgs // int(1.0/ratio)

            # sampler
            new_image_path.extend(self.image_path[begin_index:begin_index+_sample_number])
            new_gt_path.extend(self.gt_path[begin_index:begin_index+_sample_number])
         
            new_city_nubmer_imgs[name_key] = num_imgs
            new_city_cumsum[name_key] = accumsum
            accumsum += num_imgs
        self.image_path = new_image_path
        self.gt_path = new_gt_path
        self._city_number_imgs = new_city_nubmer_imgs
        self._city_cumsum = new_city_cumsum
    def _uniform_sample(self,ratio):
        cumsum, city_list = self._get_cumsum_list_and_cityname()
        new_image_path, new_gt_path = [], []
        new_city_nubmer_imgs, new_city_cumsum = {}, {}

        accumsum = 0
        for _index, begin_index in enumerate(cumsum):
            name_key = city_list[_index]
            num_imgs = self._city_number_imgs[name_key]
            _sample_number = num_imgs // int(1.0/ratio)

            # sampler
            _sampled_index = np.random.choice([i for i in range(num_imgs)],_sample_number)
            for SI in _sampled_index:
                new_image_path.append(self.image_path[SI])
                new_gt_path.append(self.gt_path[SI])
         
            new_city_nubmer_imgs[name_key] = num_imgs
            new_city_cumsum[name_key] = accumsum
            accumsum += num_imgs

        self.image_path = new_image_path
        self.gt_path = new_gt_path
        self._city_number_imgs = new_city_nubmer_imgs
        self._city_cumsum = new_city_cumsum
    def _get_cumsum_list_and_cityname(self):
        name_list = []
        number_list = []
        for index, name in enumerate(self._city_cumsum):
            number = self._city_cumsum[name]
            number_list.append(number)
            name_list.append(name)
        cumsum = np.array(number_list)
        city_list = np.array(name_list)
        index = np.argsort(cumsum)
        cumsum = cumsum[index]
        city_list = city_list[index]
        return cumsum, city_list

if __name__=="__main__":
    try_the_datasets()
