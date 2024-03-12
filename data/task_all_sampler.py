from collections import defaultdict

import numpy as np
import torch
from torch.utils.data.sampler import Sampler


class TaskAllSampler(Sampler):
    """Sampler that

    Args:
        dataset (torch.utils.data.Dataset): Dataset used for sampling.
        sampler_opt (dict): Configuration for dataset. It constains:
            num_classes (int): Number of classes for training.
            num_samples (int): Number of samples per training class
    """

    def __init__(self, dataset, sampler_opt):
        import torch.multiprocessing
        torch.multiprocessing.set_sharing_strategy('file_system')

        self.dataset = dataset
        self.num_classes = sampler_opt['num_classes']
        self.num_samples = sampler_opt['num_samples']

        self.index_dic = defaultdict(list)
        for index, label in enumerate(self.dataset.label_list):
            self.index_dic[label].append(index)



        for key in self.index_dic.keys():
            self.index_dic[key] = torch.tensor(self.index_dic[key])
        self.label_list = list(self.index_dic.keys())

        self.total_samples_per_class = len(self.index_dic[self.label_list[0]])

        self.iteration = int(self.total_samples_per_class / self.num_samples)

    def __len__(self):
        return self.num_samples * self.num_classes * self.iteration

    def __iter__(self):
        # sample self.num_classes classes from dataset
        batch = []
        classes_set = torch.randperm(len(self.index_dic))[:self.num_classes]
        samples_set  = torch.randperm(self.total_samples_per_class)
        for i in range(self.iteration):
            for cl in classes_set:
                class_label = self.label_list[cl]
                index_list = self.index_dic[class_label]
                pos = samples_set[i * self.num_samples : (i + 1) * self.num_samples].tolist()
                batch.append(index_list[pos])
        batch = torch.stack(batch).reshape(-1)
        return iter(batch)
