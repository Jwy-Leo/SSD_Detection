import torch
import os
import cv2
import torch.utils.data as data
import numpy as np
import glob2
import logging
import pickle
from torch.utils.data import DataLoader

def main():
    image_path = "/media/wy_disk/ChenYen/VIRAT/video_to_frame/VIRAT_S_000001"
    anno_path = "/media/wy_disk/ChenYen/VIRAT/annotation/anno_pickle"
    # dataset = VIRAT_Dataset(image_path = image_path, anno_path = anno_path)
    dataset = VIRAT_table_comm(image_path = image_path, anno_path = anno_path)
    labeled, unlabeled = dataset.dataset_information()
    # import pdb;pdb.set_trace()
    dataset.setting_be_selected_sample([1,3,5,7,9])
    labeled, unlabeled = dataset.dataset_information()
    # Dataloader
    DL = DataLoader(dataset,batch_size = 5,shuffle=False,num_workers = 0)
    # Change property
    dataset.Active_mode()
    # img, bb, labels = dataset.__getitem__()
    for index, (img, bbox, labels) in enumerate(DL):
        print(labels)
    # import pdb;pdb.set_trace()
    
    

class VIRAT_Dataset(data.Dataset):
    
    def __init__(self, image_path, anno_path, transform=None, target_transform=None, transforms=None, downpurning_ratio = 1.0):
        super(VIRAT_Dataset, self).__init__()
        # check if img and anno path exist
        self._check_path_exists(image_path, anno_path)
        # set up label info
        self._class_names = {u"Background":0, u"unknown":1, u"person":2, u"car":3, u"vehicles":4, u"object":5, u"bike,bicycles":6}
        self.class_names = ["Background", "unknown","person","car","vehicles","object","bike,bicycles"]
        # get data
        self.img__list = glob2.glob(image_path+"/*.png")
        num_list = [ "{:05}".format(int(str.split(str.split(img_item,"/")[-1],"_")[-1][:-4])) for img_item in self.img__list]
        
        sorted_path = sorted(zip(self.img__list,num_list), key = lambda x :x[1])
        # remove_index = -1
        # for i in range(len(sorted_path)):
        #     if int(sorted_path[i][1]) == 4546:
        #         remove_index = i
        # sorted_path.pop(remove_index)
        # purning dataset
        sorted_path = [sub_path[0] for sub_path in sorted_path]
        downsample_stride = int ( 1/downpurning_ratio )
        sorted_path = [sorted_path[i] for i in range(0,len(sorted_path),downsample_stride)]
        self.img__list = sorted_path
        
        # self.img__list.remove()
        

        anno_path = os.path.join(anno_path, "VIRAT_S_000001.viratdata.objects.pickle")
        with open(anno_path, 'rb') as f:
            self.anno__list = pickle.load(f)
        self.dataset_size = len(self.img__list)#len(self.anno__list)-1
        # transforms
        self.transform = transform 
        self.transforms = transforms
        self.target_transform = target_transform
        
        sample = {}
        # for key in self.anno__list.keys():
        for name in self.img__list:
            key_num = str.split(str.split(name,"/")[-1],"_")[-1][:-4]
            key = "frame_{}".format(key_num)
            for data in self.anno__list[key]:
                if data[7]+1 not in sample.keys():
                    sample[data[7]+1] = 1
                else:
                    sample[data[7]+1] += 1
        print(sample)

        
    
    def __getitem__(self, index):
        img = self._load_img(self.img__list[index])
        bboxes_x0y0x1y1, labels = self._load_anno(self.anno__list["frame_{}".format(str.split(self.img__list[index],"_")[-1][:-4])])
        bboxes_x0y0x1y1 = bboxes_x0y0x1y1.astype(np.float32)
        # labels_o = labels
        
        # if sum(labels_o==2)!=0:
        #     print("Labels before transform{}".format(labels))
        if self.transform:
            img, bboxes_x0y0x1y1, labels = self.transform(img, bboxes_x0y0x1y1, labels)
        # if sum(labels_o==2)!=0:
        #     print("Labels after transform{}".format(labels))
        if self.target_transform:
            bboxes_x0y0x1y1, labels = self.target_transform(bboxes_x0y0x1y1, labels)
        # if sum(labels_o==2)!=0:
        #     print("Labels after traget transform 2:{}".format(sum(labels==2)))
        # labels = labels.type('torch.LongTensor')
        
        return img, bboxes_x0y0x1y1, labels
    
    def __len__(self):
        return self.dataset_size
    
    
    def _check_path_exists(self, image_path, anno_path):
        print(image_path)
        assert os.path.exists(image_path), 'The folder path : {} is wrong '.format(image_path)
        assert os.path.exists(anno_path), 'The gound truth path : {} is wrong '.format(anno_path)
    
    def _load_img(self, img_path):
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def _load_anno(self, anno_path):
        bboxes = []
        labels = []
        for sets in anno_path:
            # import pdb;pdb.set_trace()
            x0,y0 = sets[3], sets[4]
            x1,y1 = sets[3]+sets[5], sets[4]+sets[6]
            bboxes.append(np.array([x0,y0,x1,y1]))
            labels.append(sets[7]+1)   # move the label by one because we add the background and set it to 0
        
        bboxes = np.array(bboxes) # [num_of_bboxes, x0y0x1y1]
        labels = np.array(labels) # [labels_classes, ]
        return bboxes, labels


class VIRAT_table_comm(VIRAT_Dataset):
    
    def __init__(self,image_path, anno_path, transform = None, 
                 target_transform = None, transforms = None, 
                 _sampling_mode = None, downpurning_ratio = 1.0):
        input_args = (image_path, anno_path, transform, target_transform, transforms, downpurning_ratio)
        super(VIRAT_table_comm, self).__init__(*input_args)
        self.active_mode = False
        self.reset_label_unlabel_set()
        
        
    def __getitem__(self,index):
        
        # if it is in active mode we select data from unlabeled set i.e. when we use some method to decide which data to label
        # otherwise we select from labeled set i.e. in training mode
        if self.active_mode:
            real_index = list(self.unlabel_set)[index]
        else:
            real_index = list(self.label_set)[index]
         
        img = self._load_img(self.img__list[real_index])
        bboxes_x0y0x1y1, labels = self._load_anno(self.anno__list["frame_{}".format(str.split(self.img__list[real_index],"_")[-1][:-4])])
        bboxes_x0y0x1y1 = bboxes_x0y0x1y1.astype(np.float32)
        if self.transform:
            img, bboxes_x0y0x1y1, labels = self.transform(img, bboxes_x0y0x1y1, labels)
        if self.target_transform:
            bboxes_x0y0x1y1, labels = self.target_transform(bboxes_x0y0x1y1, labels)
        # labels = labels.type('torch.LongTensor')
        
        return img, bboxes_x0y0x1y1, labels
        
    
    def Active_mode(self):
        self.active_mode = True
    
    def training_mode(self):
        self.active_mode = False
    
    # using this we can return the list and we can use np.random.choice for random
    # and use this list on sequential choicing
    def dataset_information(self):
        label_index = list(self.label_set)
        unlabel_index = list(self.unlabel_set)
        return label_index, unlabel_index
    
    # using set difference and set union to allocate different data after knowing 
    # which to put in lable or unlabel
    def setting_be_selected_sample(self, selected_sample):
        test_set = set(self.unlabel_set) - set(selected_sample)
        if len(test_set) == len(self.unlabel_set):
            print("Warning, doesn't have any sample be selected")
        self.unlabel_set = self.unlabel_set - set(selected_sample)
        self.label_set = self.label_set.union( set(selected_sample) )
    
    # using set we can use set difference and union to decide what to stay in label set and unlable set
    def reset_label_unlabel_set(self):        
        self.label_set = set()
        self.unlabel_set = set([i for i in range(len(self.img__list))])
        
    # output different len of different set we can clearly know
    # which set to fectch the data
    def __len__(self):
        if self.active_mode:
            return len(self.unlabel_set)
        else:
            return len(self.label_set)


if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
