from .vision import VisionDataset
# from ..utils.box_utils import center_form_to_corner_form
import torch
from PIL import Image
import os
import os.path
import logging
import numpy as np
import cv2
class CocoDetection(VisionDataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None):
        super(CocoDetection, self).__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.ids = self._id_filter(self.ids)
        
        label_file_name = "models/coco_label.txt"
        if os.path.isfile(label_file_name):
            #class_string = ""
            class_index = []
            with open(label_file_name, 'r') as infile:
                class_array = infile.readlines()
                class_array = [_class_item[:-1] for _class_item in class_array]
                class_index = [int(str.split(_class_item," ")[0]) for _class_item in class_array]
                class_array = [" ".join(str.split(_class_item," ")[1:]) for _class_item in class_array]

            # classes should be a comma separated list
            
            # prepend BACKGROUND as first class
            class_array.insert(0, 'BACKGROUND')
            class_index.insert(0, 0)
            self.class_index = {index_map:index for index,index_map in enumerate(class_index)}
            #classes  = [ elem.replace(" ", "") for elem in classes]
            classes = class_array
            self.class_names = tuple(classes)
            logging.info("COCO Labels read from file: " + str(self.class_names))

        else:
            logging.info("No labels file, using default VOC classes.")
            self.class_names = ('BACKGROUND',
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor')
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img = self.get_image(index)
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        target = self._hard_case_filter(target)
        boxes,labels = self.get_boxes(target)
       
        path = coco.loadImgs(img_id)[0]['file_name']

        # img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img = cv2.imread(os.path.join(self.root, path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            # img, target = self.transforms(img, target)
            # img, boxes, labels = self.transform(np.array(img), boxes, labels)
            img, boxes, labels = self.transform(img, boxes, labels)
            # img = self.transforms(img)
        
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        
        # import pdb;pdb.set_trace()
        # return img, target
        return img, boxes, labels
    def __len__(self):
        return len(self.ids)
    def get_boxes(self,target):
        boxes, labels = [], []
        for _box_id in range(len(target)):
            location_x1y1wh_form = np.array(target[_box_id]['bbox'])
            location_corner_form = np.concatenate([location_x1y1wh_form[..., :2],
                     location_x1y1wh_form[..., :2] + location_x1y1wh_form[..., 2:]],0)
            boxes.append(location_corner_form[None,:])
            labels.append(target[_box_id]['category_id'])
        boxes = np.concatenate(boxes,axis=0)
        boxes = boxes.astype('float32')
        labels = [self.class_index[labels[i]] for i in range(len(labels))]
        labels = np.array(labels)
        return boxes,labels
    def get_image(self,index):
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        img = cv2.imread(os.path.join(self.root, path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    def get_annotation(self,index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        target = self.coco.loadAnns(ann_ids)
        target = self._hard_case_filter(target)
        boxes,labels = self.get_boxes(target)
        is_difficult = np.zeros_like(labels)
        return img_id, (boxes,labels,is_difficult) 
        #return index,(boxes,labels,is_difficult) 
    def _id_filter(self,ids):
        filter_bank = []
        # Filter zero bbox images
        for i in range(len(self.ids)):
            img_id = ids[i]
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            target = self.coco.loadAnns(ann_ids)
            if len(target)==0:
                filter_bank.append(ids[i])

        ids = list( set(ids).difference(set(filter_bank)) )
        
        return ids
    def _hard_case_filter(self,target):
        remove_item = []
        for i in range(len(target)):
            if target[i]['iscrowd'] == 1:
                remove_item.append(target[i])
        for _item in remove_item:
            target.remove(_item)
        return target
    def _balance_data(self):
        label_image_indexes = [set() for _ in range(len(self.class_names))]
        for i, image in enumerate(self.data):
            for label_id in image['labels']:
                label_image_indexes[label_id].add(i)
        label_stat = [len(s) for s in label_image_indexes]
        self.min_image_num = min(label_stat[1:])
        sample_image_indexes = set()
        for image_indexes in label_image_indexes[1:]:
            image_indexes = np.array(list(image_indexes))
            sub = np.random.permutation(image_indexes)[:self.min_image_num]
            sample_image_indexes.update(sub)
        sample_data = [self.data[i] for i in sample_image_indexes]
        return sample_data

















# import numpy as np
# import logging
# import pathlib
# import xml.etree.ElementTree as ET
# import cv2
# import os


# class COCODataset:

#     def __init__(self, root, transform=None, target_transform=None, is_test=False, keep_difficult=False, label_file=None):
#         """Dataset for VOC data.
#         Args:
#             root: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
#                 Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
#         """
#         self.root = pathlib.Path(root)
#         #self.root = os.path.join("/media/disk3/wenyen/Thesis_Detection","data/pascal/VOCdevkit/VOC2012/")
#         self.root = os.path.join("/mnt/work/detection_models/Thesis_Detection","data/pascal/VOCdevkit/VOC2012/")
#         #self.root = root
#         self.transform = transform
#         self.target_transform = target_transform
#         if is_test:
#             image_sets_file = os.path.join(self.root,"ImageSets/Main/val.txt")
#         else:
#             image_sets_file = os.path.join(self.root,"ImageSets/Main/train2.txt")
        
#         self.ids = VOCDataset._read_image_ids(image_sets_file)
#         self.keep_difficult = keep_difficult

#         label_file_name = os.path.join("/media/disk3/wenyen/Thesis_Detection","data/pascal/VOCdevkit/VOC2012/ImageSets/Main/label.txt")
#         if os.path.isfile(label_file_name):
#             class_string = ""
#             with open(label_file_name, 'r') as infile:
#                 for line in infile:
#                     class_string += line.rstrip()

#             # classes should be a comma separated list
            
#             classes = class_string.split(',')
#             # prepend BACKGROUND as first class
#             classes.insert(0, 'BACKGROUND')
#             classes  = [ elem.replace(" ", "") for elem in classes]
#             self.class_names = tuple(classes)
#             logging.info("VOC Labels read from file: " + str(self.class_names))

#         else:
#             logging.info("No labels file, using default VOC classes.")
#             self.class_names = ('BACKGROUND',
#             'aeroplane', 'bicycle', 'bird', 'boat',
#             'bottle', 'bus', 'car', 'cat', 'chair',
#             'cow', 'diningtable', 'dog', 'horse',
#             'motorbike', 'person', 'pottedplant',
#             'sheep', 'sofa', 'train', 'tvmonitor')


#         self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

#     def __getitem__(self, index):
#         image_id = self.ids[index]
#         boxes, labels, is_difficult = self._get_annotation(image_id)
#         if not self.keep_difficult:
#             boxes = boxes[is_difficult == 0]
#             labels = labels[is_difficult == 0]
#         image = self._read_image(image_id)
#         if self.transform:
#             image, boxes, labels = self.transform(image, boxes, labels)
#         if self.target_transform:
#             boxes, labels = self.target_transform(boxes, labels)
#         return image, boxes, labels

#     def get_image(self, index):
#         image_id = self.ids[index]
#         image = self._read_image(image_id)
#         if self.transform:
#             image, _ = self.transform(image)
#         return image

#     def get_annotation(self, index):
#         image_id = self.ids[index]
#         return image_id, self._get_annotation(image_id)

#     def __len__(self):
#         return len(self.ids)

#     @staticmethod
#     def _read_image_ids(image_sets_file):
#         ids = []
#         with open(image_sets_file) as f:
#             for line in f:
#                 ids.append(line.rstrip())
#         return ids

#     def _get_annotation(self, image_id):
#         pass
#         return (np.array(boxes, dtype=np.float32),
#                 np.array(labels, dtype=np.int64),
#                 np.array(is_difficult, dtype=np.uint8))

#     def _read_image(self, image_id):
#         image_file = self.root + "JPEGImages/{}.jpg".format(image_id)
#         image = cv2.imread(str(image_file))
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         return image



