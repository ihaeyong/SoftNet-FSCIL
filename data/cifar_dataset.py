import sys
import os
from os import path as osp
from collections import defaultdict

import torch
import pickle

from copy import deepcopy
import numpy as np
from torch.utils import data as data

from PIL import Image, ImageEnhance
from torchvision import transforms

class TransformLoader:
    def __init__(self, transformer_opt):
        self.transformer_opt = transformer_opt

    def parse_transform(self, transform_type, transform_opt):
        method = getattr(transforms, transform_type)
        method = method(**transform_opt)
        return method

    def get_composed_transform(self):
        transform_funcs = []
        for transform_opt in self.transformer_opt:
            transform_type = transform_opt.pop('type')
            transform_funcs.append(self.parse_transform(transform_type, transform_opt))
        transform = transforms.Compose(transform_funcs)
        return transform

class CifarDataset(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
          dataroot (str): Data root path
          selected_classes: The list of classes using for training the tasks
          aug (bool): Whether proceeding data augmentation
    """

    def __init__(self, opt):
        super(CifarDataset, self).__init__()
        self.opt = opt
        self.dataroot = opt['dataroot']
        self.session = opt['session']

        self.opt_transformer_agu = deepcopy(self.opt.get('transformer_agu'))
        self.opt_transformer = deepcopy(self.opt.get('transformer'))

        self.root = osp.dirname(self.dataroot)

        meta_path = osp.join(self.root, 'meta')

        txt_path = osp.join(self.root, 'index_list', f'session_{self.session + 1}.txt')

        # split the dataset according to the order of labels. 0~59 refer to the base classes.
        self.data = []
        self.label_list = []
        with open(self.dataroot, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            self.data.append(entry['data'])
            if 'labels' in entry:
                self.label_list.extend(entry['labels'])
            else:
                self.label_list.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self.label_list = np.asarray(self.label_list)

        index = open(txt_path).read().splitlines()

        self.data, self.label_list = self.DataSelector(self.data, self.labels_list, index)

        self.softmax_label_list = self.label_list

        self.transformer_agu = None
        self.transformer_test = None
        # define the transform function
        if self.opt_transformer_agu is not None:
            tf_aug = TransformLoader(self.opt_transformer_agu)
            self.transformer_agu = tf_aug.get_composed_transform()
        if self.opt_transformer is not None:
            tf = TransformLoader(self.opt_transformer)
            self.transformer_test = tf.get_composed_transform()

        with open(meta_path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data['fine_label_names']
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

        label_unique = np.unique(self.label_list)
        self._num_classes = len(label_unique)

        self.set_aug(opt['aug'])


    def set_aug(self, aug):
        self._aug = aug
        if self._aug:
            self.transform = self.transformer_agu
        else:
            self.transform = self.transformer_test

    def get_aug(self):
        return self._aug

    # def set_buffer(self, feat_list, label_list):
    #     """
    #     set the features pre-calculated in advance
    #     :param feat_list:
    #     :param label_list:
    #     :return:
    #     """
    #     self.pre_load = True
    #     self.feat_list = feat_list.tolist()
    #     self.label_list = label_list.tolist()

    def get_num_classes(self):
        return self._num_classes

    # def sample_the_training_data(self, task_id, shots):
    #     if task_id > 0:
    #         self._sample_data(shots)

    # def sample_the_buffer_data(self, num_samples_per_class):
    #     self._sample_data(num_samples_per_class)

    def sample_the_buffer_data_with_index(self, index_list):
        self.data = [self.data[index] for index in index_list]
        self.label_list = [self.label_list[index] for index in index_list]
        self.softmax_label_list = [self.softmax_label_list[index] for index in index_list]

    def sample_all_data(self, agu=True, agu_times=1):
        if agu and agu_times > 1:
            index_list = list(range(len(self.label_list))) * agu_times
        else:
            index_list = range(len(self.label_list))
        return self.obtainImageWithIndexList(index_list, agu=agu)

    def combine_another_dataset(self, dataset):
        # TODO
        pass

        self.data = self.data + dataset.data
        self.label_list = self.label_list + dataset.label_list
        self.softmax_label_list += dataset.softmax_label_list
        self._num_classes = len(np.unique(self.label_list))
        self.relative_label_dict = {**self.relative_label_dict, **dataset.relative_label_dict}
        if self.image_list is not None and dataset.image_list is not None:
            self.image_list = self.image_list + dataset.image_list
        elif self.image_list is None and dataset.image_list is None:
            return
        else:
            raise ValueError('The origin dataset and the dataset to be combined should all have or have not image_list')

    def _sample_data(self, num_samples_per_class):

        training_data_index = self.sample_data_index(num_samples_per_class)

        self.sample_the_buffer_data_with_index(training_data_index)

    def sample_data_index(self, num_samples_per_class):
        if self._num_classes * num_samples_per_class > len(self.label_list):
            raise ValueError(f'Can not sample data from the dataset '
                             f'which only contains {len(self.label_list)} samples')

        training_data_index = []
        index_dic = defaultdict(list)
        for index, label in enumerate(self.label_list):
            index_dic[label].append(index)
        for label, index_list in index_dic.items():
            pos = torch.randperm(len(index_list))[:num_samples_per_class].tolist()
            training_data_index.extend([index_list[p] for p in pos])

        return training_data_index

    def __getitem__(self, index):
        label = self.label_list[index]
        label_softmax = self.softmax_label_list[index]

        img = self.data[index]

        image = self.transform(img)

        return (image, label, label_softmax)

    def __len__(self):
        return len(self.label_list)
