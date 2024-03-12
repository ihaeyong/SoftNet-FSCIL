from collections import defaultdict

import numpy as np
import torch
from torch.utils.data.sampler import Sampler


class MetaTaskBatchSampler(Sampler):
    """Sampler that

    Args:
        dataset (torch.utils.data.Dataset): Dataset used for sampling.
        sampler_opt (dict): Configuration for dataset. It constains:
            num_classes (int): Number of classes for training.
            num_samples (int): Number of samples per training class
    """

    def __init__(self, dataset, sampler_opt):
        self.dataset = dataset
        self.num_classes = sampler_opt['num_classes']
        self.num_samples = sampler_opt['num_samples']
        self.n_batch = sampler_opt['num_batch']
        self.total_support = sampler_opt['total_support']
        self.total_query = sampler_opt['total_query']
        self.num_support_classes = sampler_opt['num_support_classes']
        self.num_query_classes = sampler_opt['num_query_classes']


        self.index_dic = defaultdict(list)
        for index, label in enumerate(self.dataset.label_list):
            self.index_dic[label].append(index)

        for key in self.index_dic.keys():
            self.index_dic[key] = torch.tensor(self.index_dic[key])

        self.label_list = list(self.dataset.selected_classes)
        self.support_label_list = self.label_list[:self.total_support]
        self.query_label_list = self.label_list[self.total_support:]

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            # sample self.num_classes classes from dataset
            batch = []
            # random select the classes in the dataset
            support_classes_set = torch.randperm(len(self.support_label_list))[:self.num_support_classes]
            query_classes_set = torch.randperm(len(self.query_label_list))[:self.num_query_classes]

            for cl in support_classes_set:
                class_label = self.support_label_list[cl]
                index_list = self.index_dic[class_label]
                pos = torch.randperm(len(index_list))[:self.num_samples].tolist()
                batch.append(index_list[pos])

            for cl in query_classes_set:
                class_label = self.query_label_list[cl]
                index_list = self.index_dic[class_label]
                pos = torch.randperm(len(index_list))[:self.num_samples].tolist()
                batch.append(index_list[pos])

            batch = torch.stack(batch).reshape(-1)
            yield batch
