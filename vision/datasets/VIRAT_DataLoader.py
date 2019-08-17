import torch
import os
import sys
import numpy as np
import pickle
import torch.utils.data as data
import glob2
import logging
import cv2

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class VIRAT_Loader(data.Dataset):
    
    def __init__(self,image_path, anno_path, transform=None, target_transform=None, transforms=None):
        super(VIRAT_Loader,self).__init__()
        # check if img and anno path exist
        self._check_path_exists(image_path, anno_path)
        # define classes
        self._class_names = {u"BackGround":0, u"unknown":1, u"person":2, u"car":3, u"vehicles":4, u"object":5, u"bike,bicycles":6}
        self.class_names = ["BackGround","unknown","person","car","vehicles","object","bike,bicycles"]
        
        self.data_size = None
        self.img_pickle_list = glob2.glob(image_path+"/*.pickle")
        self.anno_pickle_list = glob2.glob(anno_path+"/*.pickle")
        self.transform = transform 
        self.transforms = transforms
        self.target_transform = target_transform
        self.count = 0
        # https://www.twblogs.net/a/5c835297bd9eee35fc13bd96 
        # https://blog.csdn.net/u012436149/article/details/78545766

        # Test one batch size 
        with open(self.img_pickle_list[0], 'rb') as f:
            shuffled_img = pickle.load(f) 
        self.data_size = shuffled_img.shape[0]


    def __getitem__(self, index):
        
        # if self.count == 0:
        #     self.shuffled_img = self._load_samples()
        #     self.BBOXes, self.LABELes = self._load_anno()    
        # else:
        #     pass        
        # img = self.shuffled_img[index]
        # bboxes_x0y0x1y1, labels = self.BBOXes[index], self.LABELes[index]
        # if self.transform:
        #     img, bboxes_x0y0x1y1, labels = self.transform(self.shuffled_img[index], self.BBOXes[index], self.LABELes[index])
        # if self.target_transform:
        #     bboxes_x0y0x1y1, labels = self.target_transform(self.BBOXes[index], self.LABELes[index])
        #     img = self.shuffled_img[index]
        img = self._load_samples(index)
        bboxes_x0y0x1y1, labels = self._load_anno(index)    
        
        logging.debug("===== before transform VIRAT img shape : {} ======".format(img.shape))
        logging.debug("===== before transform VIRAT bbox shape : {} & type : {} ======".format(bboxes_x0y0x1y1.shape, bboxes_x0y0x1y1.dtype))
        logging.debug("===== before transform VIRAT labels shape : {} & type : {} ======".format(labels.shape, labels.dtype))
        
        if self.transform:
            img, bboxes_x0y0x1y1, labels = self.transform(img, bboxes_x0y0x1y1, labels)
        if self.target_transform:
            bboxes_x0y0x1y1, labels = self.target_transform(bboxes_x0y0x1y1, labels)
        labels = labels.type('torch.LongTensor')
        logging.debug("===== VIRAT img shape : {} ======".format(img.shape))
        logging.debug("===== VIRAT bbox shape : {} ======".format(bboxes_x0y0x1y1.shape))
        logging.debug("===== VIRAT label shape : {} ======".format(labels.shape))
        
        return img, bboxes_x0y0x1y1, labels
  
    def __len__(self):
        # return self.data_size
        return self.data_size * len(self.img_pickle_list)

    def _check_path_exists(self, image_path, anno_path):
        print(image_path)
        assert os.path.exists(image_path), 'The folder path : {} is wrong '.format(image_path)
        assert os.path.exists(anno_path), 'The gound truth path : {} is wrong '.format(anno_path)
    
    def _load_samples(self, index):
        fetch_data_pickle = index // self.data_size
        fetch_data_slice = index % self.data_size
        if int(sys.version[0]) > 2:
            with open(self.img_pickle_list[fetch_data_pickle], 'rb') as f:
                shuffled_img = pickle.load(f) 
        else:
            with open(self.img_pickle_list[fetch_data_pickle], 'rb') as f:
                raise NotImplementedError("Can't load by python 2")
                import pdb;pdb.set_trace()
                shuffled_img = pickle.load(f, encoding = 'latin1') 
            
        # self.img_pickle_list.append(self.img_pickle_list[0])
        # del self.img_pickle_list[0]
        # original shape is (N.C,H,W) change to (N,W,H,C)
        # shuffled_img = shuffled_img.transpose(0,3,2,1)
        # self.data_size = shuffled_img.shape[0]
        # self.count = 1
        shuffled_img = shuffled_img[fetch_data_slice,...]
        shuffled_img = shuffled_img.transpose(1,2,0)
        shuffled_img = shuffled_img.astype(np.uint8)
        shuffled_img = cv2.cvtColor(shuffled_img, cv2.COLOR_BGR2RGB)
        
        return shuffled_img
    
    def _load_anno(self, index):
        fetch_data_pickle = index // self.data_size
        fetch_data_slice = index % self.data_size
        
        with open(self.anno_pickle_list[fetch_data_pickle], 'rb') as f:
            shuffled_anno = pickle.load(f)
        # self.anno_pickle_list.append(self.anno_pickle_list[0])
        # del self.anno_pickle_list[0]
        # shuffled_anno is a list
        # inside the list is a array
        # batch_size = len(shuffled_anno)
        shuffled_anno = shuffled_anno[fetch_data_slice]
        
        # BBOXes = []
        # Labels = []
        # for each_b in shuffled_anno:
        #     bboxes = []
        #     labels = []
        #     for sets in each_b:
        #         x0,y0 = sets[3], sets[4]
        #         x1,y1 = sets[3]+sets[5], sets[4]+sets[6]
        #         bboxes.append(np.array([x0,y0,x1,y1]))
        #         labels.append(sets[7])
        #     BBOXes.append(np.array(bboxes))
        #     Labels.append(np.array(labels))
        # BBOXes = np.array(BBOXes)  # [batchsize, bbox_of_each_frame, x0y0x1y1]
        # LABELes = np.array(Labels)  # [batchsize, labels_of_each_frame, label_classes] 

        
        bboxes = []
        labels = []
        for sets in shuffled_anno:
            x0,y0 = sets[3], sets[4]
            x1,y1 = sets[3]+sets[5], sets[4]+sets[6]
            bboxes.append(np.array([x0,y0,x1,y1]))
            labels.append(sets[7])
        
        BBOXes = np.array(bboxes)  # [batchsize, bbox_of_each_frame, x0y0x1y1]
        LABELes = np.array(labels)  # [batchsize, labels_of_each_frame, label_classes] 
        logging.debug("========= BBOXes shape:{} =======".format(BBOXes.shape))
        logging.debug("========= LABELes shape:{} =======".format(LABELes.shape))
        return BBOXes, LABELes
    

        
        
        