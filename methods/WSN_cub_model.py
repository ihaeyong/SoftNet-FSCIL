import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
import os
import wandb
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from methods import networks as networks
from methods.base_model import BaseModel
from methods.mt_nb_model import MTNBModel
from utils import ProgressBar, get_root_logger, Averager, dir_size, mkdir_or_exist, pnorm, AvgDict
from data.normal_dataset import NormalDataset
from data import create_sampler, create_dataloader, create_dataset
from metrics import pair_euclidean_distances, pair_euclidean_distances_dim3
from metrics.norm_cosine_distances import pair_norm_cosine_distances, pair_norm_cosine_distances_dim3, norm_mahalanobis, pair_norm_pearson_distances

import numpy as np
from methods.optim.utils import group_product, group_add, normalization, get_params_grad, hessian_vector_product, orthnormal

from methods.optim.sgd import SGD
from methods.optim.adam import Adam

from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

loss_module = importlib.import_module('methods.losses')

class WSNCubModel(BaseModel):
    """Metric-based with random noise to parameters learning model"""
    def __init__(self, opt, sess_id=0):
        super(WSNCubModel, self).__init__(opt)

        self.use_cosine = self.opt.get('use_cosine', False)
        self.device = self.opt.get('gpu', False)

        if self.is_incremental:
            train_opt = self.opt['train']
            self.now_session_id = self.opt['task_id'] + 1
            self.num_novel_class = train_opt['num_class_per_task'] if self.now_session_id > 0 else 0
            self.total_class = train_opt['bases'] + self.num_novel_class * self.now_session_id
            self.num_old_class = self.total_class - self.num_novel_class if self.now_session_id > 0 else self.total_class
        else:
            self.now_session_id = 0

        # define network
        opt_net = deepcopy(opt['network_g'])
        self.finetune_layer = self.opt.get('finetune_layer', None)
        self.net_g = networks.define_net_g(opt_net)
        self.net_g = self.model_to_device(self.net_g, self.device)
        if not (self.is_incremental and self.now_session_id > 0):
            self.print_network(self.net_g)

        # set sparsity
        sparsity = opt['network_g']['sparsity']
        self.net_g.func.apply(lambda m: setattr(m, "sparsity", sparsity))

        # set softnet
        #if self.opt['subnet_type'] == 'softnet' and self.is_incremental:
        #self.net_g.func.apply(lambda m: setattr(m, "smooth", True))
        #else:
        #    self.net_g.func.apply(lambda m: setattr(m, "smooth", False))

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_model_g', None)
        if load_path is not None:
            if self.opt['datasets']['train']['name'] == 'cub':
                self.load_base_network(self.net_g, load_path,
                                   self.opt['path']['strict_load'])
            else:
                self.load_network(self.net_g, load_path,
                                  self.opt['path']['strict_load'])

        # load subnet masks and update consolidated masks
        load_base_subnet_path = self.opt['path'].get('base_subnet_model', None)
        #self.per_task_masks = torch.load(load_base_subnet_path)
        if load_base_subnet_path is not None:
            print(f' --------------------------------------------------------')
            print(f' ---- load subnet_file:{load_base_subnet_path}')
            print(f' --------------------------------------------------------')
            self.per_task_masks = torch.load(load_base_subnet_path)
        else:
            self.per_task_masks = {}
            self.per_task_masks[0] = None

        # load base models for incremental learning
        load_base_model_path = self.opt['path'].get('base_model', None)
        if load_base_model_path is not None and self.is_incremental:
            self.load_network(self.net_g, load_base_model_path,
                              self.opt['path']['strict_load'],
                              is_train=True)


            # load the prototypes for all seen classes
            assert opt['task_id'] + 1 == self.now_session_id
            self.load_prototypes(opt['task_id'], opt['test_id'])

            # record the former network
            self.net_g_former = deepcopy(self.net_g)
            self.net_g_former.eval()

        self.init_training_settings()

    def incremental_init(self, train_set, val_set=None):
        """ Initializing the incremental learning procedure
        Args:
            train_set (torch.utils.data.Dataset): the training dataset
            val_set (torch.utils.data.Dataset): the validation dataset
        """
        self.novel_classes = train_set.targets

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        # define losses
        self.loss_func = nn.CrossEntropyLoss()

        if train_opt.get('cpn_opt'):
            cf_type = train_opt['cpn_opt'].pop('type')
            pn_loss_func = getattr(loss_module, cf_type)
            self.cpn_loss_func = pn_loss_func(**train_opt['cpn_opt']).to(self.device)
            train_opt['cpn_opt']['type'] = cf_type
        else:
            self.cpn_loss_func = None

        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        lr_cf = train_opt['optim_g'].get('lr_cf', None)

        logger = get_root_logger()
        if lr_cf is not None:
            train_opt['optim_g'].pop('lr_cf')
            opitm_embed = []
            optim_cf = []
            for k, v in self.net_g.named_parameters():
                if v.requires_grad:
                    if 'fc' in k:
                        logger.warning(f'FC Params {k} will be optimized.')
                        optim_cf.append(v)
                    else:
                        logger.warning(f'Params {k} will be optimized.')
                        opitm_embed.append(v)
                else:
                    logger.warning(f'Params {k} will not be optimized.')

            optim_params = [{'params': opitm_embed},
                            {'params': optim_cf, 'lr': lr_cf}]
        else:
            optim_params = []
            for k, v in self.net_g.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger = get_root_logger()
                    logger.warning(f'Params {k} will not be optimized.')

        self.optim_type = train_opt['optim_g'].pop('type')
        if self.optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params,
                                                **train_opt['optim_g'])
        elif self.optim_type == 'SGD':
            self.optimizer_g = torch.optim.SGD(optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)
        train_opt['optim_g']['type'] = self.optim_type

    def incremental_optimize_parameters(self, current_iter, mask=None):
        # set zero grads
        self.optimizer_g.zero_grad()

        model = getattr(self.net_g, 'func')
        model.train()

        # get per task masks
        if self.opt['subnet_type'] == 'softnet':
            per_new_task = self.get_masks(1, sparsity = 1-self.opt['network_g']['sparsity'],
                                          layer = self.finetune_layer)

        # SubNet w/o per_task_mask for feature diversity
        per_task_mask = self.per_task_masks[0] if True else None
        output_new_task, output_new_task_fc = model.forward_score(
                self.images, mask=per_task_mask, mode='train')

        l_total = 0
        if self.cpn_loss_func is not None :
            loss, log = self.cpn_loss_func(self.former_proto_list.detach(),
                                           self.former_proto_label.detach(),
                                           output_new_task,
                                           self.labels)

            self.log_dict.update(log)
            self.log = log
            l_total += loss


        # update minor params
        if self.opt['subnet_type'] == 'softnet':
            l_total.backward(retain_graph=False)
            key_list = self.per_task_masks[0].keys()

            target_layer_list=[]
            for key in key_list:
                if self.opt['finetune_layer'] in key and 'bias' not in key:
                    target_layer_list.append(key)

            print(f' ---- target_layer:{target_layer_list}')
            for key, param in model.named_parameters():

                if not param.requires_grad:
                    continue

                if 'w_m' in key :
                    param.grad = None

                if key in key_list:
                    m = self.grad_mask[key].detach()
                    mv = self.old_mask[key].detach()
                    # only target layer should be updated
                    if key in target_layer_list:
                        if True:
                            # soft-scaled gradient
                            param.grad *= m * mv
                        else:
                            # hard-scaled gradient
                            param.grad *= m
                    else:
                        param.grad = None

            # update only minor params
            self.optimizer_g.step()

    def incremental_update(self, novel_dataset, mask=None):
        train_opt = self.opt['val']
        test_type = train_opt.get('test_type', 'NCM')

        if mask is None:
            mask = self.per_task_masks[0]

        if test_type == 'NCM' or self.now_session_id == 0:
            prototypes_list, labels_list = self.get_prototypes(novel_dataset, mask)
            if labels_list.max() > 60:
                # update prototypes dict
                for i in range(prototypes_list.shape[0]):
                    self.prototypes_dict.update({labels_list[i].item(): prototypes_list[i]})

            else:
                # update prototypes dict
                for i in range(prototypes_list.shape[0]):
                    self.prototypes_dict.update({labels_list[i].item(): prototypes_list[i]})

        print("----- len_of_prototypes:{}".format(self.prototypes_dict.keys()))

    def incremental_test(self, test_dataset, task_id=-1, test_id=-1, mask=None):
        self.net_g.eval()
        train_opt = self.opt['val']

        if mask is None :
            mask = self.per_task_masks[0]

        test_type = train_opt.get('test_type', 'NCM')
        if test_type == 'NCM' or self.now_session_id == 0:
            acc = self.__NCM_incremental_test(test_dataset, task_id, test_id, mask=mask)
        else:
            raise ValueError(f'Do not support the type {test_type} for incremental testing!')

        return acc

    def incremental_fine_tune(self, train_loader, val_dataset, num_epoch, task_id=-1, test_id=-1, tb_logger=None, mask=None):
        """
        fine tune the models with the samples of incremental novel class

        Args:
            train_dataset (torch.utils.data.Dataset): the training dataset
            val_dataset (torch.utils.data.Dataset): the validation dataset
            num_epoch (int): the number of epoch to fine tune the models
            task_id (int): the id of sessions
            test_id (int): the id of few-shot test
        """

        self.task_id = task_id

        current_iter = 0
        for epoch in range(num_epoch):
            self.epoch = epoch
            for idx, data in enumerate(train_loader):
                current_iter += 1
                self.update_learning_rate(
                    current_iter, warmup_iter=-1)

                #self.tr_labels.append(data[1])

                with torch.no_grad():
                    self.feed_data(data)

                self.incremental_optimize_parameters(current_iter, mask=mask)

                print('epoch:{}, iter:{}'.format(epoch, current_iter))

    def percentile(self, scores, sparsity):
        k = 1 + round(.01 * float(sparsity) * (scores.numel() - 1))
        return scores.view(-1).kthvalue(k).values.item()

    def smooth(self, scores, onehot, gamma=0.9, tau=0.3):
        atten = F.softmax(scores.view(-1) / tau, dim=0).view(scores.size())
        smooth = gamma * onehot + (1-gamma) * atten
        return smooth

    def get_soft_mask(self, mask, sparsity=0.3, layer='layer4', scale=1.0):
        '''
        model         : base + shared + new
        mask          : base + shared
        new_task_mask : new  + shared
        base_mask     : mask - shared1
        new           : mask - shared
        '''
        model = self.net_g_former.func
        sparsity -= 0.0001  # for numerical stability
        new_task_mask = deepcopy(mask)
        new_mask = deepcopy(mask)
        old_mask = deepcopy(mask)

        # avoid overlapped weights
        # SoftNet
        new_sparsity = 0.99 # 100 % re-used
        old_sparsity = 0.01
        for key in self.per_task_masks[0].keys():

            if 'bias' in key:
                continue

            if layer not in key:
                continue

            new_mask[key] = torch.zeros_like(mask[key])
            old_mask[key] = torch.zeros_like(mask[key])

            key_split = key.split('.')
            if len(key_split) == 2:
                module_attr = 'w_m'
                module_name = key_split[0]

                if (hasattr(getattr(model, module_name), module_attr)):
                    if (getattr(getattr(model, module_name), module_attr) is not None):
                        w_m = getattr(getattr(model, module_name), module_attr)

                        # acquire one-hot mask
                        ones = torch.ones_like(w_m)
                        zeros = torch.zeros_like(w_m)

                        #k_val = self.percentile(mask[key], sparsity * 100)
                        #h_mask = torch.where(mask[key] < k_val, ones, zeros)

                        k_val = self.percentile(mask[key][h_mask==1], new_sparsity * 100)
                        new_m = torch.where(mask[key][h_mask==1] < k_val, ones, zeros)

                        # get new weights
                        ones = torch.ones_like(w_m[h_mask==1])
                        zeros = torch.zeros_like(w_m[h_mask==1])

                        # update per_task_mask
                        new_task_m = torch.zeros_like(w_m)

                        # less c% weights
                        #k_val = self.percentile(w_m[h_mask==1], new_sparsity * 100)
                        #new_m = torch.where(w_m[h_mask==1] <= k_val, ones, zeros)
                        k_val = self.percentile(mask[key][h_mask==1], new_sparsity * 100)
                        new_m = torch.where(mask[key][h_mask==1] < k_val, ones, zeros)
                        # update per_task_mask
                        if True:
                            new_task_m[h_mask==0] = mask[key][h_mask==0]
                            new_task_m[h_mask==1] = mask[key][h_mask==1]
                        else:
                            new_task_m[h_mask==1] = new_m * 1.0
                            new_task_m[h_mask==0] = 1.0

                            new_task_m=self.smooth(w_m, new_task_m,
                                                   gamma=0.9,tau=0.3)

                        # get new weights
                        ones = torch.ones_like(w_m[h_mask==0])
                        zeros = torch.zeros_like(w_m[h_mask==0])

                        #k_val = self.percentile(w_m[h_mask==0], old_sparsity * 100)
                        #old_m = torch.where(w_m[h_mask==0] <= k_val, ones, zeros)

                        k_val = self.percentile(mask[key][h_mask==0], old_sparsity * 100)
                        old_m = torch.where(mask[key][h_mask==0] < k_val, ones, zeros)

                        new_task_mask[key] = new_task_m
                        new_mask[key][h_mask==1] = new_m
                        old_mask[key][h_mask==0] = old_m

            else:
                module_attr = 'w_m'

                # layer conv
                if len(key_split) == 4:
                    curr_module = getattr(getattr(model, key_split[0])[int(key_split[1])], key_split[2])

                # downsample
                elif len(key_split) == 5:
                    curr_module = getattr(getattr(model, key_split[0])[int(key_split[1])], key_split[2])[int(key_split[3])]

                if hasattr(curr_module, module_attr):
                    if getattr(curr_module, module_attr) is not None:

                        w_m = getattr(curr_module, module_attr)

                        # acquire one-hot mask
                        ones = torch.ones_like(w_m)
                        zeros = torch.zeros_like(w_m)

                        k_val = self.percentile(mask[key], sparsity * 100)
                        h_mask = torch.where(mask[key] < k_val, ones, zeros)

                        # get new weights
                        ones = torch.ones_like(w_m[h_mask==1])
                        zeros = torch.zeros_like(w_m[h_mask==1])

                        # update per_task_mask
                        new_task_m = torch.zeros_like(w_m)

                        # less c% weights
                        #k_val = self.percentile(w_m[h_mask==1].abs(), new_sparsity * 100)
                        #new_m = torch.where(w_m[h_mask==1].abs() < k_val, ones, zeros)

                        k_val = self.percentile(mask[key][h_mask==1], new_sparsity * 100)
                        new_m = torch.where(mask[key][h_mask==1] < k_val, ones, zeros)

                        # update per_task_mask
                        if True:
                            new_task_m[h_mask==0] = mask[key][h_mask==0]
                            new_task_m[h_mask==1] = mask[key][h_mask==1]
                        else:
                            new_task_m[h_mask==1] = new_m * 1.0
                            new_task_m[h_mask==0] = 1.0

                            new_task_m=self.smooth(w_m,new_task_m,
                                                   gamma=0.9,tau=0.3)

                        # get new weights
                        ones = torch.ones_like(w_m[h_mask==0])
                        zeros = torch.zeros_like(w_m[h_mask==0])

                        #k_val = self.percentile(w_m[h_mask==0], old_sparsity * 100)
                        #old_m = torch.where(w_m[h_mask==0] <= k_val, ones, zeros)
                        k_val = self.percentile(mask[key][h_mask==0], old_sparsity * 100)
                        old_m = torch.where(mask[key][h_mask==0] < k_val, ones, zeros)

                        new_task_mask[key] = new_task_m
                        new_mask[key][h_mask==1] = new_m
                        old_mask[key][h_mask==0] = old_m

        return new_task_mask, new_mask, old_mask

    def get_curr_masks(self, sess_id):
        per_task_mask = deepcopy(self.net_g.get_masks(sess_id))
        return per_task_mask

    def get_masks(self, sess_id, sparsity=0.3, layer='layer3'):

        if sess_id > 0:
            per_new_task, self.grad_mask, self.old_mask = self.get_soft_mask(
                mask=self.per_task_masks[0], sparsity=sparsity, layer=layer)
        elif sess_id == -1:
            per_new_task = self.get_curr_masks(0)
        else:
            per_new_task = self.per_task_masks[0]

        return per_new_task

    def log_dist(self, prototypes):
        sim = pair_norm_cosine_distances(self.output,
                                         prototypes,
                                         mode='train')
        n_base_cls = self.opt['train']['bases']
        former, f_idx = sim[0,:n_base_cls].topk(1)
        former = former.mean()
        if sim.shape[1] > n_base_cls:
            novel, n_idx = sim[0,n_base_cls:].topk(1)
            novel = novel.mean()
            dist_proto = pair_norm_cosine_distances(
                prototypes[f_idx,:],
                prototypes[n_idx,:], mode='train')
            dist_proto = dist_proto.mean().item()
        else:
            novel = former * 0.0
            dist_proto = 0.0

        # dist former and novel
        dist_fn = pair_norm_cosine_distances(self.output,
                                             self.output_bcls,
                                             mode='train')

        alpha = np.exp(-np.abs(former.item()-novel.item()))

        return former, novel, alpha, dist_proto, dist_fn

    def __NCM_incremental_test(self, test_dataset, task_id=-1, test_id=-1, mask=None, tb_logger=None):
        prototypes = []
        pt_labels = []
        p_norm = self.opt['val'].get('p_norm', None)
        for key, value in self.prototypes_dict.items():
            prototypes.append(value)
            pt_labels.append(key)
        prototypes = torch.stack(prototypes).to(self.device)
        pt_labels = torch.tensor(pt_labels).to(self.device)

        data_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  drop_last=False)

        acc_ave = Averager()
        acc_former_ave = Averager()
        acc_former_all_ave = Averager()
        acc_novel_all_ave = Averager()

        # test norm
        novel_norm = Averager()
        old_norm = Averager()

        if p_norm is not None and self.now_session_id > 0:
            prototypes = pnorm(prototypes, p_norm)

        sim_former_list = []
        sim_novel_list = []
        dist_proto_list = []
        dist_former_novel_list = []
        pred_list = []
        alpha_list = []
        for idx, data in enumerate(data_loader):
            self.feed_data(data)
            self.test(mask=mask)

            if self.use_cosine:
                feature = self.output
                pairwise_distance = pair_norm_cosine_distances(feature, prototypes)
                estimate = torch.argmin(pairwise_distance, dim=1)
            else:
                feature = self.output
                pairwise_distance = pair_euclidean_distances(feature,
                                                             prototypes)
                estimate = torch.argmin(pairwise_distance, dim=1)

            estimate_labels = pt_labels[estimate]
            acc = (estimate_labels ==
                   self.labels).sum() / float(estimate_labels.shape[0])

            acc_ave.add(acc.item(), int(estimate_labels.shape[0]))
            pred_list.append(estimate_labels.item())

            if self.opt.get('details', False):
                if self.labels.item() < self.opt['train']['bases']:
                    acc_former_all_ave.add(acc.item(), int(estimate_labels.shape[0]))
                else:
                    acc_novel_all_ave.add(acc.item(), int(estimate_labels.shape[0]))

        # tentative for out of GPU memory
        del self.images
        del self.labels
        del self.output
        torch.cuda.empty_cache()

        if self.opt.get('details', False):
            log_str = f'[Test_acc of task {task_id} on test {test_id}: {acc_ave.item():.5f}]' \
                      f'[acc of former classes: {acc_former_ave.item():.5f}]' \
                      f'[acc of former samples in all classes: {acc_former_all_ave.item():.5f}]\n' \
                      f'[acc of novel samples in all classes: {acc_novel_all_ave.item():.5f}]'
                      # f'[old norm: {old_norm.item():.5f}][novel norm: {novel_norm.item():.5f}]'
        else:
            log_str = f'[Test_acc of task {task_id} on test {test_id}: {acc_ave.item():.5f}]'

        logger = get_root_logger()
        logger.info(log_str)

        if self.opt.get('details', False):
            return acc_ave.item(), acc_former_ave.item(), acc_former_all_ave.item(), acc_novel_all_ave.item()
        else:
            return acc_ave.item()


    def feed_data(self, data):
        """
        The Data structure is (images, labels, labels_softmax)
        """
        self.images = data[0].to(self.device)
        self.labels = data[1].to(self.device)
        try:
            self.labels_softmax = data[2].to(self.device)
        except:
            self.labels_softmax = data[1].to(self.device)

    def clear_grad(self):
        """
        Clear the grad on all the parameters
        """
        for p in self.net_g.parameters():
            p.grad = None

    def test(self, mask=None, mode='test'):
        self.net_g.eval()
        with torch.no_grad():
            if mask is not None:
                per_task_mask = mask
            else:
                per_task_mask = self.per_task_masks[0]
            # shared weight + new
            self.output, self.output_fc = self.net_g.forward_score(
                self.images,
                mask=per_task_mask,
                mode=mode)

    def optimize_parameters(self, current_iter, mask=None, smooth=False):
        self.optimizer_g.zero_grad()

        self.net_g.train()
        # ========== base learning ==========
        if smooth:
            self.net_g.func.apply(lambda m: setattr(m, "smooth", True))
        else :
            self.net_g.func.apply(lambda m: setattr(m, "smooth", False))

        self.score, original_output = self.net_g.forward_score(self.images,
                                                               mask = None,
                                                               mode = 'train')

        self.log_dict = AvgDict()
        loss = self.loss_func(original_output, self.labels_softmax)

        log_dict = {'CELoss': loss.item()}
        self.log_dict.add_dict(log_dict)
        loss.backward()
        self.optimizer_g.step()

        self.log_dict = self.log_dict.get_ordinary_dict()

    def get_prototypes(self, training_dataset, mask=None):
        """
        calculated the prototypes for each class in training dataset
        Args:
            training_dataset (torch.utils.data.Dataset): the training dataset

        Returns:
            tuple: (prototypes_list, labels_list) where prototypes_list is the list of prototypes and
            labels_list is the list of class labels
        """
        features_list = []
        labels_list = []
        prototypes_list = []
        data_loader = torch.utils.data.DataLoader(
            training_dataset, batch_size=128, shuffle=False,
            drop_last=False)

        for i, data in enumerate(data_loader, 0):
            self.feed_data(data)
            self.test(mask=mask)

            feature = self.output
            features_list.append(feature)
            labels_list.append(self.labels)

        # tentative for out of GPU memory
        del self.images
        del self.labels
        del self.output
        torch.cuda.empty_cache()

        features = torch.cat(features_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        selected_classes = np.unique(training_dataset.targets)
        print('----selected classes:{}'.format(selected_classes))
        for cl in selected_classes:
            index_cl = torch.where(cl == labels)[0]
            class_features = features[index_cl]
            if self.use_cosine:
                class_features = F.normalize(class_features, dim=1)
            prototypes_list.append(class_features.mean(dim=0))

        prototypes_list = torch.stack(prototypes_list, dim=0)
        return prototypes_list, selected_classes
    
    def nondist_validation(self, training_dataset, dataloader, current_iter, tb_logger, name='', mask=None):
        """
        Args:
            current_iter: the current iteration. -1 means the testing procedure
        """
        self.net_g.eval()

        if mask is None :
            mask = self.per_task_masks[self.now_session_id]

        acc = self.__nondist_validation(training_dataset, dataloader, mask=mask)
        log_str = f'Val_acc \t {acc:.5f}\n'
        logger = get_root_logger()
        logger.info(log_str)

        if current_iter != -1:
            if tb_logger:
                tb_logger.add_scalar(f'{name}val_acc', acc, current_iter)
            if self.wandb_logger is not None:
                wandb.log({f'{name}val_acc': acc}, step=current_iter)
        else:
            if tb_logger:
                tb_logger.add_scalar(f'{name}val_acc', acc, 0)
            if self.wandb_logger is not None:
                wandb.log({f'{name}val_acc': acc}, step=0)


        return acc

    def __nondist_validation(self, training_dataset, dataloader, mask=None):
        acc_ave = Averager()

        for idx, data in enumerate(dataloader):
            self.feed_data(data)
            self.test(mask=mask, mode='test')

            estimate_labels = torch.argmax(self.output_fc, dim=1)
            acc = (estimate_labels ==
                   self.labels_softmax).sum() / float(estimate_labels.shape[0])

            acc_ave.add(acc.item(), int(self.labels_softmax.shape[0]))

        # tentative for out of GPU memory
        del self.images
        del self.labels
        del self.labels_softmax
        del self.output

        return acc_ave.item()

    def save(self, epoch, current_iter, name='net_g', dataset=None,
             mask=None, task_id=None):
        self.save_network(self.net_g, name, current_iter)
        self.save_mask(mask, task_id, name, current_iter)
        self.save_training_state(epoch, current_iter)
        if self.is_incremental:
            self.save_prototypes(self.now_session_id, self.opt['test_id'])

    def save_prototypes(self, session_id, test_id):
        if session_id >= 0:
            save_path = osp.join(self.opt['path']['prototypes'], f'test{test_id}_session{session_id}.pt')
            torch.save(self.prototypes_dict, save_path)

    def load_prototypes(self, session_id, test_id):
        if session_id >= 0:
            if self.opt['train']['novel_exemplars'] == 0:
                load_filename = f'test{0}_session{session_id}.pt'
            else:
                # for topic split
                load_filename = f'test{0}_session{session_id}.pt'

            load_path = osp.join(self.opt['path']['prototypes'], load_filename)
            logger = get_root_logger()
            logger.info(f'Load prototypes: {load_path}')
            prototypes_dict = torch.load(load_path)
            self.prototypes_dict = prototypes_dict
            self.former_proto_list, self.former_proto_label = self._read_prototypes()
        else:
            if self.opt['path'].get('pretrain_prototypes', None) is not None:
                prototypes_dict = torch.load(self.opt['path']['pretrain_prototypes'])
                self.prototypes_dict = prototypes_dict
                self.former_proto_list, self.former_proto_label = self._read_prototypes()

    def set_the_saving_files_path(self, opt, task_id):
        # change the path of base model
        save_filename_g = f'session_{task_id}.pth'
        save_path_g = osp.join(opt['path']['models'], save_filename_g)
        opt['path']['base_model'] = save_path_g

    def _get_features(self, dataset):
        aug = dataset.get_aug()
        dataset.set_aug(False)

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=8, shuffle=False, drop_last=False)

        features = []
        labels = []
        for i, data in enumerate(data_loader, 0):
            self.feed_data(data)
            self.test()
            features.append(self.output.cpu())
            labels.append(self.labels.cpu())

        del self.images
        del self.labels
        del self.output
        torch.cuda.empty_cache()

        dataset.set_aug(aug)

        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)
        return features, labels

    def _read_prototypes(self):
        prototypes = []
        pt_labels = []
        for key, value in self.prototypes_dict.items():
            prototypes.append(value)
            pt_labels.append(key)
        if len(prototypes) > 0:
            prototypes = torch.stack(prototypes).to(self.device)
            pt_labels = torch.tensor(pt_labels).to(self.device)
        else:
            prototypes = None
            pt_labels = None
        return prototypes, pt_labels

