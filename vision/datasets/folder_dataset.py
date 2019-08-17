import numpy as np
import logging
import xml.etree.ElementTree as ET
import cv2
import os
import pickle
import json

def main():
    folder_path = "/media/wy_disk/wy_file/Detection/dataset/datasets/ECP_Golden_pattern"
    D = FolderDataset(folder_path)
    for i in range(len(D)):
        image = D.get_image(i)
        label = D.get_annotation(i)
        
class FolderDataset(object):
    def __init__(self, root, transform=None, target_transform=None):
        image_path = os.path.join(root,"images")
        mfolder, _folders, _files = next(iter(os.walk(image_path)))
        num_list = [ "{:05}".format(int(str.split(_file,"_")[-1][:-4])) for _file in _files]
        
        sorted_path = sorted(zip(_files,num_list), key = lambda x :x[1])
        sorted_path = [sub_path[0] for sub_path in sorted_path]
        _files = sorted_path
        # _files = sorted(_files)
        self.image_pth_list = [os.path.join(mfolder,_file) for _file in _files]
        self.ids = [i for i in range(len(self.image_pth_list))]
        
        label_path = os.path.join(root,"labels")
        # mfolder, _folders, _files = next(iter(label_path))
        # _files = sorted(_files)
        #self.label_pth_list = [os.path.join(label_path,str.split(_file,".")[0]+".json") for _file in _files]
        self.label_path = os.path.join(label_path,"VIRAT_S_000001.viratdata.objects.pickle")
        with open(self.label_path, 'rb') as F:
            self.label_dict = pickle.load(F)
        self.transform = transform
        self.target_transform = target_transform

    def get_image(self, index):
        image_id = self.ids[index]
        image = self._read_image(image_id)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        image_id = self.ids[index]
        image_key = str.split(str.split(self.image_pth_list[image_id],"/")[-1],".")[0]
        return image_id, self._get_annotation(image_key)

    def __len__(self):
        return len(self.ids)
    
    def _read_image(self,image_id):
        img_path = self.image_pth_list[image_id]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        return img
    def _get_annotation(self,image_id):
        #self.label_path[]
        labels = self.label_dict[image_id]
        data_temp = []
        gt = []
        for label in labels:
            data_temp.append([label[3], label[4], label[3]+label[5],label[4]+label[6]])
            gt.append(label[-1]+1)
        data_temp = np.vstack(data_temp)
        gt = np.vstack(gt)
        return data_temp, gt
        


if __name__=="__main__":
    main()
