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
from utils import ProgressBar, get_root_logger, Averager, dir_size, mkdir_or_exist, pnorm
from data.normal_dataset import NormalDataset
from data import create_sampler, create_dataloader, create_dataset
from metrics import pair_euclidean_distances, pair_euclidean_distances_dim3
from metrics.norm_cosine_distances import pair_norm_cosine_distances, pair_norm_cosine_distances_dim3, norm_mahalanobis, pair_norm_pearson_distances

import numpy as np
#from methods.optim.hessian import hessian
from methods.optim.utils import group_product, group_add, normalization, get_params_grad, hessian_vector_product, orthnormal

from methods.optim.sgd import SGD
#from methods.optim.ada_hessian import AdaHessian
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

loss_module = importlib.import_module('methods.losses')

class WSNFusionModel(BaseModel):
    """Metric-based with random noise to parameters learning model"""
    def __init__(self, opt, sess_id=0):
        super(WSNFusionModel, self).__init__(opt)

        self.use_cosine = self.opt.get('use_cosine', False)
        self.device = self.opt.get('gpu', False)

        if self.is_incremental:
            train_opt = self.opt['train']
            self.now_session_id = self.opt['task_id'] + 1
            self.num_novel_class = train_opt['num_class_per_task'] if self.now_session_id > 0 else 0
            self.total_class = train_opt['bases'] + self.num_novel_class * self.now_session_id
            self.num_old_class = self.total_class - self.num_novel_class if self.now_session_id > 0 else self.total_class

        # define network
        opt_net = deepcopy(opt['network_g'])
        self.finetune_layer = self.opt.get('finetune_layer', None)
        self.net_g = networks.define_net_g(opt_net)
        self.net_g = self.model_to_device(self.net_g, self.device)
        if not (self.is_incremental and self.now_session_id > 0):
            self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_model_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path']['strict_load'])

        # load subnet masks and update consolidated masks
        load_base_subnet_path = self.opt['path'].get('base_subnet_model', None)
        self.per_task_masks = torch.load(load_base_subnet_path)

        # load base models for incremental learning
        load_base_model_path = self.opt['path'].get('base_model', None)
        if load_base_model_path is not None and self.is_incremental:
            self.load_network(self.net_g, load_base_model_path,
                              self.opt['path']['strict_load'], is_train=True)
            # load the prototypes for all seen classes
            self.load_prototypes(opt['task_id'], opt['test_id'])

            # record the former network
            self.net_g_former = deepcopy(self.net_g)
            self.net_g_former.eval()

        # set sparsity
        sparsity = opt['network_g']['sparsity']
        self.net_g.func.apply(lambda m: setattr(m, "sparsity", sparsity))

        if opt['subnet_type'] == 'softnet':
            self.net_g.func.apply(lambda m: setattr(m, "smooth", True))
        else:
            self.net_g.func.apply(lambda m: setattr(m, "smooth", False))

        if self.is_train or (self.is_incremental and self.opt['train']['fine_tune']):
             self.init_training_settings()

        # total classes
        self.image_list = []
        self.label_list = []
        self.output_list = []

        # base classes
        self.label_bcls_list = []
        self.output_bcls_list = []

        # new incremental classes
        self.label_new_list = []
        self.output_new_list = []

        self.tr_labels = []
        self.te_labels = []

        self.new2base = []
        self.new2new =[]
        self.base2base = []
        self.base2new = []

        self.dist_b2b = []
        self.dist_n2b = []
        self.margin = []

        self.bias_base = []
        self.bias_new = []

        self.base_mask = None
        self.grad_mask = None
        self.output_base = None
        self.output_new = None

    def incremental_init(self, train_set, val_set):
        """ Initializing the incremental learning procedure
        Args:
            train_set (torch.utils.data.Dataset): the training dataset
            val_set (torch.utils.data.Dataset): the validation dataset
        """
        self.novel_classes = train_set.selected_classes

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        # define losses
        self.loss_func = nn.CrossEntropyLoss()
 
        # define losses
        if train_opt.get('metric_opt'):
            metric_type = train_opt['metric_opt'].pop('type')
            metric_loss_func = getattr(loss_module, metric_type)
            self.metric_loss_func = metric_loss_func(**train_opt['metric_opt']).to(self.device)
            train_opt['metric_opt']['type'] = metric_type
        else:
            self.metric_loss_func = None

        if train_opt.get('pn_opt'):
            cf_type = train_opt['pn_opt'].pop('type')
            pn_loss_func = getattr(loss_module, cf_type)
            self.pn_loss_func = pn_loss_func(**train_opt['pn_opt']).to(self.device)
            train_opt['pn_opt']['type'] = cf_type
        else:
            self.pn_loss_func = None

        if train_opt.get('cpn_opt'):
            cf_type = train_opt['cpn_opt'].pop('type')
            pn_loss_func = getattr(loss_module, cf_type)
            self.cpn_loss_func = pn_loss_func(**train_opt['cpn_opt']).to(self.device)
            train_opt['cpn_opt']['type'] = cf_type
        else:
            self.cpn_loss_func = None

        # regularization
        if train_opt.get('regularization'):
            regular_type = train_opt['regularization'].pop('type')
            regularization_func = getattr(loss_module, regular_type)
            self.regularization_func = regularization_func(**train_opt['regularization']).to(self.device)
            train_opt['regularization']['type'] = regular_type
        else:
            self.regularization_func = None

        self.freeze_networks_with_threshold(train_opt['threshold'])

        self.setup_optimizers()
        self.setup_schedulers()

    def mask_diff(self, mask):
        for key in self.per_task_masks[0].keys():
            if mask[key] is not None:
                diff = mask[key] - self.per_task_masks[0][key]
                if diff.abs().sum() > 0:
                    print("key:{}, diff:{}".format(key, diff.abs().sum()))

    def incremental_fc_optimize_parameters(self, current_iter, mask=None):

        # set zero grads
        self.optimizer_g.zero_grad()

        model = getattr(self.net_g, 'func')
        model_former = getattr(self.net_g_former, 'func')
        model.train()
        model_former.eval()

        # get per task masks
        if self.opt['subnet_type'] == 'softnet':
            per_new_task = self.get_masks(1, sparsity = 1-self.opt['network_g']['sparsity'],
                                          layer = self.finetune_layer)

        # SubNet w/o per_task_mask for feature diversity
        per_task_mask = self.per_task_masks[0] if False else None
        output_new_task, output_new_task_fc = model.forward_score(
                self.images, mask=per_task_mask, mode='train')

        output_base, output_new = None, None

        l_total = 0
        if self.cpn_loss_func is not None :
            loss, log = self.cpn_loss_func(self.former_proto_list.detach(),
                                           self.former_proto_label.detach(),
                                           output_new_task,
                                           output_base,
                                           output_new,
                                           self.labels)
            self.log_dict.update(log)
            l_total += loss

        if self.opt['subnet_type'] == 'softnet':
            l_total.backward(retain_graph=False)

            mask_list = []
            mask_v_list = []
            param_list = []
            key_list = self.per_task_masks[0].keys()
            for key, param in model.named_parameters():

                if not param.requires_grad:
                    continue

                if 'w_m' in key :
                    m = self.grad_mask[key.replace('w_m', 'weight')].detach()
                    mv = self.old_mask[key.replace('w_m', 'weight')].detach()
                    if self.opt['finetune_layer'] in key :
                        mask_list.append(m)
                        mask_v_list.append(mv)
                    else:
                        mask_list.append(torch.zeros_like(m))
                        mask_v_list.append(torch.zeros_like(mv))
                param_list.append(key)

                # mask should not updated
                if key in key_list:
                    m = self.grad_mask[key].detach()
                    mv = self.old_mask[key].detach()
                    #if self.opt['finetune_layer'] in key :
                    #    mask_list.append(m)
                    #    mask_v_list.append(mv)
                    #else:
                    mask_list.append(torch.zeros_like(m))
                    mask_v_list.append(torch.zeros_like(mv))
                param_list.append(key)

            self.optimizer_g.step(mask=mask_list, mask_v=mask_v_list)

    def incremental_update(self, novel_dataset, mask=None):
        train_opt = self.opt['val']
        test_type = train_opt.get('test_type', 'NCM')

        if mask is None or self.task_id == 1:
            mask = self.per_task_masks[0]

        if test_type == 'NCM' or self.now_session_id == 0:
            prototypes_list, labels_list = self.get_prototypes(novel_dataset, mask)

            # update prototypes dict
            for i in range(prototypes_list.shape[0]):
                self.prototypes_dict.update({labels_list[i].item(): prototypes_list[i]})

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

    def incremental_fine_tune(self, train_dataset, val_dataset, num_epoch, task_id=-1, test_id=-1, tb_logger=None, mask=None):
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
        sampler_opt = self.opt['datasets']['train']['sampler']
        sampler_opt['num_classes'] = self.num_novel_class

        train_sampler = create_sampler(train_dataset, sampler_opt)
        dataset_opt = self.opt['datasets']['train']

        train_loader = create_dataloader(
            train_dataset,
            dataset_opt,
            sampler=train_sampler,
            seed=self.opt['manual_seed'])

        current_iter = 0
        for epoch in range(num_epoch):
            self.epoch = epoch
            for idx, data in enumerate(train_loader):
                current_iter += 1
                self.update_learning_rate(
                    current_iter, warmup_iter=-1)

                self.tr_labels.append(data[2])

                with torch.no_grad():
                    self.feed_data(data)

                self.incremental_fc_optimize_parameters(current_iter, mask=mask)

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
        new_task_mask = deepcopy(mask)
        new_mask = deepcopy(mask)
        old_mask = deepcopy(mask)

        # avoid overlapped weights
        sparsity -= 0.0001
        new_sparsity = 0.99 # 100 % re-used ratio
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

                        k_val = self.percentile(mask[key], sparsity * 100)
                        h_mask = torch.where(mask[key] < k_val, ones, zeros)

                        # get new weights
                        ones = torch.ones_like(w_m[h_mask==1])
                        zeros = torch.zeros_like(w_m[h_mask==1])

                        # update per_task_mask
                        new_task_m = torch.zeros_like(w_m)

                        k_val = self.percentile(mask[key][h_mask==1], new_sparsity * 100)
                        new_m = torch.where(mask[key][h_mask==1] < k_val, ones, zeros)

                        # update per_task_mask
                        new_task_m[h_mask==0] = mask[key][h_mask==0]
                        new_task_m[h_mask==1] = mask[key][h_mask==1]

                        # get new weights
                        ones = torch.ones_like(w_m[h_mask==0])
                        zeros = torch.zeros_like(w_m[h_mask==0])

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

                        k_val = self.percentile(mask[key][h_mask==1], new_sparsity * 100)
                        new_m = torch.where(mask[key][h_mask==1] < k_val, ones, zeros)

                        # update per_task_mask
                        new_task_m[h_mask==0] = mask[key][h_mask==0]
                        new_task_m[h_mask==1] = mask[key][h_mask==1]

                        # get new weights
                        ones = torch.ones_like(w_m[h_mask==0])
                        zeros = torch.zeros_like(w_m[h_mask==0])

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

        if len(self.tr_labels) > 0:
            self.tr_labels = np.concatenate(self.tr_labels)

            print("==task_id:{}, test_id:{} w/ train labels{}:from {} to {}".format(
                task_id, test_id, len(self.tr_labels), self.tr_labels.min(), self.tr_labels.max()))
        print("==task_id:{}, test_id:{} w/ test labels{}:from {} to {}".format(
            task_id, test_id, prototypes.size(), pt_labels.min(), pt_labels.max()))

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

        if p_norm is not None :
            prototypes = pnorm(prototypes, p_norm)

        sim_former_list = []
        sim_novel_list = []
        dist_proto_list = []
        dist_former_novel_list = []
        pred_list = []
        alpha_list = []
        for idx, data in enumerate(data_loader):
            self.feed_data(data)
            self.test()

            self.image_list.append(self.images)
            self.output_list.append(self.output)
            self.label_list.append(self.labels)

            if self.use_cosine:
                feature = self.output

                if p_norm is not None and self.now_session_id > 0:
                    feature = pnorm(feature, p_norm)

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

            if self.labels.item() < self.opt['train']['bases']:
                acc_former_all_ave.add(acc.item(), int(estimate_labels.shape[0]))
            else:
                acc_novel_all_ave.add(acc.item(), int(estimate_labels.shape[0]))

        if True:
            former, novel, alpha, dist_proto, dist_fn = self.log_dist(prototypes)
            sim_former_list.append(former.item())
            sim_novel_list.append(novel.item())
            alpha_list.append(alpha)
            dist_proto_list.append(dist_proto)
            dist_former_novel_list.append(dist_fn)

            log = {'labels': self.label_list,
                   'preds': pred_list,
                   'acc_former': acc_former_all_ave.item(),
                   'acc_novel': acc_novel_all_ave.item(),
                   'sim_former': sim_former_list,
                   'sim_novel': sim_novel_list,
                   'alpha': alpha_list,
                   'dist_proto':dist_proto_list,
                   'dist_fn': dist_former_novel_list}

            self.log_dict.update(log)

        self.save_file(test_id, task_id)
        # tentative for out of GPU memory
        del self.images
        del self.labels
        del self.output
        del self.output_bcls
        torch.cuda.empty_cache()

        log_str = f'[Test_acc of task {task_id} on test {test_id}: {acc_ave.item():.5f}]'
        logger = get_root_logger()
        logger.info(log_str)

        return acc_ave.item()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        self.param_list = []
        for k, v in self.net_g.named_parameters():

            if not v.requires_grad :
                continue

            key = k.replace('func.', '')
            if key in self.per_task_masks[0].keys():
                optim_params.append(v)
                self.param_list.append(key)

            if 'w_m' in key:
                optim_params.append(v)
                self.param_list.append(key)

            if key not in self.param_list:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params,
                                                **train_opt['optim_g'])
        elif optim_type == 'SGD':
            self.optimizer_g = torch.optim.SGD(optim_params, **train_opt['optim_g'])
        elif optim_type == 'iSGD':
            self.optimizer_g = SGD(optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdaH':
            self.optimizer_g = AdaHessian(optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)
        train_opt['optim_g']['type'] = optim_type

    def feed_data(self, data):
        """
        The Data structure is (images, labels, labels_softmax)
        """
        self.images = data[0].to(self.device)
        self.labels = data[1].to(self.device)
        self.labels_softmax = data[2].to(self.device)

    def clear_grad(self):
        """
        Clear the grad on all the parameters
        """
        for p in self.net_g.parameters():
            p.grad = None

    def test(self, mask=None, mode='test'):

        self.net_g.apply(lambda m: setattr(m, "alpha", None))
        self.net_g.eval()

        with torch.no_grad():
            # shared weight + new
            self.output, self.output_fc = self.net_g.forward_score(
                self.images,
                mask=self.per_task_masks[0],
                mode=mode)

            # shared weight + base
            self.output_bcls, self.output_bcls_fc = self.net_g_former.forward_score(
                self.images,
                mask=self.per_task_masks[0],
                mode=mode)

    def get_prototypes(self, training_dataset, mask=None):
        """
        calculated the prototypes for each class in training dataset
        Args:
            training_dataset (torch.utils.data.Dataset): the training dataset

        Returns:
            tuple: (prototypes_list, labels_list) where prototypes_list is the list of prototypes and
            labels_list is the list of class labels
        """
        aug = training_dataset.get_aug()
        training_dataset.set_aug(False)

        features_list = []
        labels_list = []
        prototypes_list = []
        data_loader = torch.utils.data.DataLoader(
            training_dataset, batch_size=128, shuffle=False, drop_last=False)

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
        selected_classes = training_dataset.selected_classes
        for cl in selected_classes:
            index_cl = torch.where(cl == labels)[0]
            class_features = features[index_cl]
            if self.use_cosine:
                class_features = F.normalize(class_features, dim=1)
            prototypes_list.append(class_features.mean(dim=0))

        prototypes_list = torch.stack(prototypes_list, dim=0)
        # reset augmentation
        training_dataset.set_aug(aug)
        return prototypes_list, torch.from_numpy(training_dataset.selected_classes)

    def dist_plot(self,test_id, task_id):
        new2base = torch.stack(self.new2base).detach().cpu().numpy()
        new2new = torch.stack(self.new2new).detach().cpu().numpy()
        base2base = torch.stack(self.base2base).detach().cpu().numpy()
        base2new = torch.stack(self.base2new).detach().cpu().numpy()
        labels = torch.cat(self.te_labels).detach().cpu().numpy()

        margin = torch.stack(self.margin).detach().cpu().numpy()

        b2b = torch.stack(self.dist_b2b).detach().cpu().numpy()
        n2b = torch.stack(self.dist_n2b).detach().cpu().numpy()

        bias_new = torch.stack(self.bias_new).detach().cpu().numpy()
        bias_base = torch.stack(self.bias_base).detach().cpu().numpy()

        feat_cols = [ 'pixel'+str(i) for i in range(len(labels)) ]

        df = pd.DataFrame(new2base, columns=['new2base'])
        df['y'] = labels
        df['label'] = df['y'].apply(lambda i: str(i))
        df['new2new'] = new2new
        df['base2base'] = base2base
        df['base2new'] = base2new
        df['b2b'] = b2b
        df['n2b'] = n2b
        df['margin'] = margin
        df['bias_new'] = bias_new
        df['bias_base'] = bias_base

        plt.figure(figsize=(32,20))
        ax_new2base = plt.subplot(2, 2, 1)
        sns.scatterplot(
            x="y", y="bias_base",
            hue="y",
            palette=sns.color_palette("hls", labels.max()+1),
            data=df,
            legend=False,
            alpha=0.3,
            ax=ax_new2base
        )

        ax_new2new = plt.subplot(2, 2, 2)
        sns.scatterplot(
            x="y", y="bias_new",
            hue="y",
            palette=sns.color_palette("hls", labels.max()+1),
            data=df,
            legend=False,
            alpha=0.3,
            ax=ax_new2new
        )

        ax_base2base = plt.subplot(2, 2, 3)
        sns.scatterplot(
            x="y", y="base2base",
            hue="y",
            palette=sns.color_palette("hls", labels.max()+1),
            data=df,
            legend=False,
            alpha=0.3,
            ax=ax_base2base
        )

        ax_base2new = plt.subplot(2, 2, 4)
        sns.scatterplot(
            x="y", y="base2new",
            hue="y",
            palette=sns.color_palette("hls", labels.max()+1),
            data=df,
            legend=False,
            alpha=0.3,
            ax=ax_base2new
        )

        plt.savefig('./tb_logger_{}/'.format(self.opt['datasets']['train']['name']) + self.opt['name'] + '/dist_sess{}_task_id{}.pdf'.format(test_id, task_id), dpi=300, format='pdf', bbox_inches='tight')
        plt.close()

    def save_file(self, test_id, task_id):

        log = {}
        #log['image'] = self.image_list
        log['output'] = self.output_list
        #log['output_base'] = self.output_bcls_list
        #log['output_new'] = self.output_new_list

        log['label'] = self.label_list
        #log['label_base'] = self.label_bcls_list
        #log['labels_new'] = self.label_new_list

        file_name = './tb_logger_{}/'.format(self.opt['datasets']['train']['name']) + self.opt['name'] + '/sess{}_task_id{}.pt'.format(test_id, task_id)

        torch.save(log, file_name)

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

            estimate_labels = torch.argmax(self.output, dim=1)
            acc = (estimate_labels ==
                   self.labels_softmax).sum() / float(estimate_labels.shape[0])

            acc_ave.add(acc.item(), int(self.labels_softmax.shape[0]))

        # tentative for out of GPU memory
        del self.images
        del self.labels
        del self.labels_softmax
        del self.output

        return acc_ave.item()

    def save(self, epoch, current_iter, name='net_g', dataset=None, mask=None, task_id=None):
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
                load_filename = f'test{0}_session{0}.pt'
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

    def freeze_networks_with_threshold(self, threshold):
        self.optim_param_name = []
        self.optim_param = []
        self.former_optim_param = []

        for key, param in self.net_g.named_parameters():
            v = param.data.abs().max().item()

            if False:
                if v > threshold:
                    param.requires_grad = False
                else:
                    if key.find('fc') == -1 and key.find('classifier') == -1:
                        self.optim_param_name.append(key)
                        self.optim_param.append(param)

        for name in self.optim_param_name:
            logger = get_root_logger()
            logger.info(f'Optimize parameters: {name}.')

        for key, param in self.net_g_former.named_parameters():
            if key in self.optim_param_name:
                self.former_optim_param.append(param)
