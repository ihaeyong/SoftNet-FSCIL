import os
import os.path as osp

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from .autoaugment import AutoAugBrightnessPolicy, AutoAugContrastPolicy, AutoAugSaturationPolicy


class CUB200(Dataset):

    def __init__(self, root='./', train=True, index_path=None, index=None, base_sess=None, two_images=False, validation=False):
        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set
        self._pre_operate(self.root)

        image_size = 224
        self.train_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # not strengthened
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # alice
        ])

        self.train_transforms_v1 = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomApply([AutoAugBrightnessPolicy()], p=0.1),    # ColorJitter
            transforms.RandomApply([AutoAugContrastPolicy()], p=0.1),      # ColorJitter
            transforms.RandomApply([AutoAugSaturationPolicy()], p=0.1),    # ColorJitter
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # alice
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.train_transform_v2 = transforms.Compose([
            transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        
        self.base_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # alice
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.base_transforms_v2 = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # alice
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.transform_test_v1 = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomApply([AutoAugBrightnessPolicy()], p=1.0),    # ColorJitter
            transforms.RandomApply([AutoAugContrastPolicy()], p=1.0),      # ColorJitter
            transforms.RandomApply([AutoAugSaturationPolicy()], p=1.0),    # ColorJitterz
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # alice
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


        if train:
            if validation:
                print('---- CUB200 Base Transform ---')
                self.transform = self.base_transforms
            elif not base_sess:
                print('---- CUB200 Base Transform ---')
                self.transform = self.base_transforms
            else:
                if two_images:
                    print('---- CUB200 TwoCrops Training Transform ---')
                    self.transform = TwoCropsTransform(train_transforms)
                else:
                    print('---- CUB200 OneCrops Training Transform ---')
                    self.transform = self.train_transforms

            if base_sess:
                self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
            else:
                self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)
        else:
            if validation:
                print('---- CUB200 Base Transform ---')
                self.transform = self.base_transforms
            else:
                print('---- CUB200 Testing Transform ---')
                self.transform = self.transform_test

            self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)

    def set_aug(self, flag=False):

        if flag:
            print('---- CUB200 OneCrops Training Transform ---')
            self.transform = self.train_transforms
        else:
            print('---- CUB200 Testing (base) Transform ---')
            self.transform = self.transform_test

    def text_read(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                lines[i] = line.strip('\n')
        return lines

    def list2dict(self, list):
        dict = {}
        for l in list:
            s = l.split(' ')
            id = int(s[0])
            cls = s[1]
            if id not in dict.keys():
                dict[id] = cls
            else:
                raise EOFError('The same ID can only appear once')
        return dict

    def _pre_operate(self, root):
        image_file = os.path.join(root, 'CUB_200_2011/images.txt')
        split_file = os.path.join(root, 'CUB_200_2011/train_test_split.txt')
        class_file = os.path.join(root, 'CUB_200_2011/image_class_labels.txt')
        id2image = self.list2dict(self.text_read(image_file))
        id2train = self.list2dict(self.text_read(split_file))  # 1: train images; 0: test iamges
        id2class = self.list2dict(self.text_read(class_file))
        train_idx = []
        test_idx = []
        for k in sorted(id2train.keys()):
            if id2train[k] == '1':
                train_idx.append(k)
            else:
                test_idx.append(k)

        self.data = []
        self.targets = []
        self.data2label = {}
        if self.train:
            for k in train_idx:
                image_path = os.path.join(root, 'CUB_200_2011/images', id2image[k])
                self.data.append(image_path)
                self.targets.append(int(id2class[k]) - 1)
                self.data2label[image_path] = (int(id2class[k]) - 1)

        else:
            for k in test_idx:
                image_path = os.path.join(root, 'CUB_200_2011/images', id2image[k])
                self.data.append(image_path)
                self.targets.append(int(id2class[k]) - 1)
                self.data2label[image_path] = (int(id2class[k]) - 1)

    def SelectfromTxt(self, data2label, index_path):
        data_tmp = []
        targets_tmp = []

        for i in range(len(index_path)):
            index = open(index_path[i]).read().splitlines()
            for i in index:
                img_path = os.path.join(self.root, i)
                data_tmp.append(img_path)
                targets_tmp.append(data2label[img_path])

        return data_tmp, targets_tmp

    def SelectfromClasses(self, data, targets, index):
        data_tmp = []
        targets_tmp = []
        for i in index:
            ind_cl = np.where(i == targets)[0]
            for j in ind_cl:
                data_tmp.append(data[j])
                targets_tmp.append(targets[j])

        return data_tmp, targets_tmp

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, targets = self.data[i], self.targets[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, targets

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


