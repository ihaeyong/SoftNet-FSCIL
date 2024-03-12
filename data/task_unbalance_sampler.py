from collections import defaultdict

import numpy as np
import torch
from torch.utils.data.sampler import Sampler


class TaskUBBatchSampler(Sampler):
    """Sampler that

    Args:
        dataset (torch.utils.data.Dataset): Dataset used for sampling.
        sampler_opt (dict): Configuration for dataset. It constains:
            num_classes (int): Number of classes for training.
            num_samples (int): Number of samples per training class
    """

    def __init__(self, dataset, sampler_opt):
        self.dataset = dataset
        num_classes = sampler_opt['n_ways']
        self.num_samples = sampler_opt['n_instances']
        self.num_base_classes = sampler_opt['num_base_classes']
        self.n_batch = sampler_opt['num_batch']
        ratio = sampler_opt['ratio']

        self.n_sampled_novel_classes = int(num_classes * ratio)
        self.n_sampled_old_classes = num_classes - self.n_sampled_novel_classes

        all_classes = dataset.all_classes
        base_classes = all_classes[:self.num_base_classes]
        self.old_index_dic = defaultdict(list)
        self.novel_index_dic = defaultdict(list)
        for index, label in enumerate(self.dataset.label_list):
            if label in base_classes:
                self.old_index_dic[label].append(index)
            else:
                self.novel_index_dic[label].append(index)

        for key in self.old_index_dic.keys():
            self.old_index_dic[key] = torch.tensor(self.old_index_dic[key])
        for key in self.novel_index_dic.keys():
            self.novel_index_dic[key] = torch.tensor(self.novel_index_dic[key])

        self.old_label_list = list(self.old_index_dic.keys())
        self.novel_label_list = list(self.novel_index_dic)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            # sample self.num_classes classes from dataset
            batch = []
            # random select the classes in the old classes
            classes_set = torch.randperm(len(self.old_index_dic))[:self.n_sampled_old_classes]
            for cl in classes_set:
                class_label = self.old_label_list[cl]
                index_list = self.old_index_dic[class_label]
                pos = torch.randperm(len(index_list))[:self.num_samples].tolist()
                batch.append(index_list[pos])

            # random select the classes in the novel classes
            classes_set = torch.randperm(len(self.novel_index_dic))[:self.n_sampled_novel_classes]
            for cl in classes_set:
                class_label = self.novel_label_list[cl]
                index_list = self.novel_index_dic[class_label]
                pos = torch.randperm(len(index_list))[:self.num_samples].tolist()
                batch.append(index_list[pos])

            batch = torch.stack(batch).reshape(-1)
            yield batch
