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
from metrics.norm_cosine_distances import pair_norm_cosine_distances, pair_norm_cosine_distances_dim3

import numpy as np
#from methods.optim.hessian import hessian
from methods.optim.utils import group_product, group_add, normalization, get_params_grad, hessian_vector_product, orthnormal

from methods.optim.sgd import SGD

loss_module = importlib.import_module('methods.losses')

class WSNModel(BaseModel):
    """Metric-based with random noise to parameters learning model"""
    def __init__(self, opt, sess_id=0):
        super(WSNModel, self).__init__(opt)

        self.use_cosine = self.opt.get('use_cosine', False)
        self.device = self.opt.get('gpu', False)

        if self.is_incremental:
            train_opt = self.opt['train']
            self.now_session_id = self.opt['task_id'] + 1
            self.num_novel_class = train_opt['num_class_per_task'] if self.now_session_id > 0 else 0
            self.total_class = train_opt['bases'] + self.num_novel_class * self.now_session_id
            self.num_old_class = self.total_class - self.num_novel_class if self.now_session_id > 0 else self.total_class

        # define network
        self.net_g = networks.define_net_g(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g, self.device)
        if not (self.is_incremental and self.now_session_id >0):
            self.print_network(self.net_g)

        self.sess_id = sess_id

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_model_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path']['strict_load'])

        # load subnet masks and update consolidated masks
        load_base_subnet_path = self.opt['path'].get('base_subnet_model', None)
        self.per_task_masks = np.load(load_base_subnet_path, allow_pickle=True).item()
        self.update_consolidated_masks()

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

        if self.is_train or (self.is_incremental and self.opt['train']['fine_tune']):
             self.init_training_settings()

        self.tr_labels = []

    def update_consolidated_masks(self):
        if True:
            self.consolidated_masks = deepcopy(self.per_task_masks[self.sess_id])
        else:
            for key in self.per_task_masks[self.sess_id].keys():
                # Or operation on sparsity
                if self.consolidated_masks[key] is not None and self.per_task_masks[self.sess_id][key] is not None:
                    self.consolidated_masks[key]=1-((1-self.consolidated_masks[key]) * (1-self.per_task_masks[self.sess_id][key]))

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

        # regularization
        if train_opt.get('regularization'):
            regular_type = train_opt['regularization'].pop('type')
            regularization_func = getattr(loss_module, regular_type)
            self.regularization_func = regularization_func(**train_opt['regularization']).to(self.device)
            train_opt['regularization']['type'] = regular_type
        else:
            self.regularization_func = None

        #self.freeze_networks_with_threshold(train_opt['threshold'])

        self.setup_optimizers()
        self.setup_schedulers()

    def consolidated_update(self, model):
        for key in self.consolidated_masks.keys():
            # Determine wheter it's an output head or not
            key_split = key.split('.')
            if len(key_split) == 2:
                module_attr = key_split[1]
                module_name = key_split[0]

                # Zero-out gradients
                if (hasattr(getattr(model, module_name), module_attr)):
                    if (getattr(getattr(model, module_name), module_attr) is not None):
                        pre_train_w = getattr(getattr(model, module_name), module_attr)[self.consolidated_masks[key]==1].mean()
                        trf_train_w = getattr(getattr(model, module_name), module_attr)[self.consolidated_masks[key]==0].mean()

                        pre_train_g = getattr(getattr(model, module_name), module_attr).grad * self.consolidated_masks[key].float()
                        trf_train_g = getattr(getattr(model, module_name), module_attr).grad * (1-self.consolidated_masks[key]).float()

                        trf_w_ratio = (self.consolidated_masks[key] == 0).sum() / self.consolidated_masks[key].view(-1).size()[0]
                        print('=========={}==========='.format(key))
                        print('epoch:{},{},trf_ratio:{}'.format(self.epoch, module_name, trf_w_ratio))
                        print('pre_train_w:{}, trf_trn_w:{}'.format(pre_train_w, trf_train_w))
                        #print('pre_train_g:{}, trf_trn_g:{}'.format(pre_train_g, trf_train_g))

                        sz = pre_train_g.size(0)
                        pre_subspace = torch.mm(pre_train_g.view(sz, -1).t(),pre_train_g.view(sz, -1))
                        trf_train_g = torch.mm(trf_train_g.view(sz, -1), pre_subspace).reshape(trf_train_g.size())
                        pre_train_g = torch.mm(pre_train_g.view(sz, -1), pre_subspace).reshape(pre_train_g.size())
                        if False:
                            getattr(getattr(model, module_name), module_attr).grad[self.consolidated_masks[key] == 1] = pre_train_g[self.consolidated_masks[key] == 1]
                            getattr(getattr(model, module_name), module_attr).grad[self.consolidated_masks[key] == 0] = trf_train_g[self.consolidated_masks[key] == 0]

                        else:
                            getattr(getattr(model, module_name), module_attr).grad[self.consolidated_masks[key] == 1] = 0
                            getattr(getattr(model, module_name), module_attr).grad[self.consolidated_masks[key] == 0] = 0


            else:
                module_attr = key_split[-1]

                # Zero-out gradients
                curr_module = getattr(getattr(model, key_split[0])[int(key_split[1])], key_split[2])
                if hasattr(curr_module, module_attr):
                    if getattr(curr_module, module_attr) is not None:
                        pre_train_w = getattr(curr_module, module_attr)[self.consolidated_masks[key]==1].mean()
                        trf_train_w = getattr(curr_module, module_attr)[self.consolidated_masks[key]==0].mean()

                        pre_train_g = getattr(curr_module, module_attr).grad * self.consolidated_masks[key].float()
                        trf_train_g = getattr(curr_module, module_attr) * (1-self.consolidated_masks[key]).float()

                        trf_w_ratio = (self.consolidated_masks[key] == 0).sum() / self.consolidated_masks[key].view(-1).size()[0]
                        print('{},trf_ratio:{}'.format(curr_module, trf_w_ratio))
                        print('pre_train_w:{}, trf_trn_w:{}'.format(pre_train_w, trf_train_w))
                        #print('pre_train_g:{}, trf_trn_g:{}'.format(pre_train_g, trf_train_g))

                        sz = pre_train_g.size(0)
                        pre_subspace = torch.mm(pre_train_g.view(sz, -1).t(),pre_train_g.view(sz, -1))
                        trf_train_g = torch.mm(trf_train_g.view(sz, -1), pre_subspace).reshape(trf_train_g.size())
                        pre_train_g = torch.mm(pre_train_g.view(sz, -1), pre_subspace).reshape(pre_train_g.size())

                        if False:
                            getattr(curr_module, module_attr).grad[self.consolidated_masks[key] == 1] = pre_train_g[self.consolidated_masks[key]==1]
                            getattr(curr_module, module_attr).grad[self.consolidated_masks[key] == 0] = trf_train_g[self.consolidated_masks[key]==0]
                        else:
                            if getattr(curr_module, module_attr).requires_grad:
                                getattr(curr_module, module_attr).grad[self.consolidated_masks[key] == 1] = 0
                                #getattr(curr_module, module_attr).grad[self.consolidated_masks[key] == 0] = 0

    def consolidated_diff(self, model):
        with torch.no_grad():
            for key in self.consolidated_masks.keys():
                key_split = key.split('.')
                if len(key_split) == 2:
                    module_attr = key_split[1]
                    module_name = key_split[0]

                    if (hasattr(getattr(model, module_name), module_attr)):
                        if (getattr(getattr(model, module_name), module_attr) is not None):
                            prev_w = getattr(getattr(self.net_g_former.func, module_name), module_attr)
                            curr_w = getattr(getattr(model, module_name), module_attr)

                            diff_w = prev_w - curr_w
                            cons_w = diff_w[self.consolidated_masks[key]==1]
                            spar_w = diff_w[self.consolidated_masks[key]==0]
                            print("key:{}, cons_w:{}, spar_w:{}".format(key, cons_w.mean(), spar_w.mean()))

                else:
                    module_attr = key_split[-1]

                    # Zero-out gradients
                    curr_module = getattr(getattr(model, key_split[0])[int(key_split[1])], key_split[2])
                    prev_module = getattr(getattr(self.net_g_former.func, key_split[0])[int(key_split[1])], key_split[2])
                    if hasattr(curr_module, module_attr):
                        if getattr(curr_module, module_attr) is not None:
                            prev_w = getattr(prev_module, module_attr)
                            curr_w = getattr(curr_module, module_attr)

                            diff_w = prev_w - curr_w

                            cons_w = diff_w[self.consolidated_masks[key]==1]
                            spar_w = diff_w[self.consolidated_masks[key]==0]

                            print("key:{}, cons_w:{}, spar_w:{}".format(key, cons_w.mean(), spar_w.mean()))

    def incremental_optimize_parameters(self, current_iter, mask=None):
        self.optimizer_g.zero_grad()
        model = getattr(self.net_g, 'func')
        model.train()
        model.zero_grad()
        output = model(self.images, mask=None,  mode='train')

        if self.loss_func:
            loss = self.loss_func(output, self.labels_softmax)

        if self.metric_loss_func:
            loss, log = self.metric_loss_func(output, self.labels)
            # update the log_dict
            self.log_dict.update(log)

        if self.pn_loss_func is not None:
            loss, log = self.pn_loss_func(self.former_proto_list.detach(), self.former_proto_label.detach(), output, self.labels)
            self.log_dict.update(log)

        if self.regularization_func is not None:
            loss, log = self.regularization_func(self.former_optim_param, self.optim_param)
            self.log_dict.update(log)

        loss.backward()

        for key in self.consolidated_masks.keys():
            key_split = key.split('.')
            if len(key_split) == 2:
                module_attr, module_name = key_split[1], key_split[0]

                # Zero-out gradients
                if hasattr(getattr(model, module_name), module_attr):
                    if getattr(getattr(model, module_name), module_attr) is not None:
                        if getattr(getattr(model, module_name), module_attr).requires_grad:
                            getattr(getattr(model, module_name), module_attr).grad[self.consolidated_masks[key] == 0] = 0
            else:
                module_attr = key_split[-1]
                # Zero-out gradients
                curr_module = getattr(getattr(model, key_split[0])[int(key_split[1])], key_split[2])
                if hasattr(curr_module, module_attr):
                    if getattr(curr_module, module_attr) is not None:
                        if getattr(curr_module, module_attr).requires_grad:
                            getattr(curr_module, module_attr).grad[self.consolidated_masks[key] == 0] = 0


        key_list = self.consolidated_masks.keys()
        # consolidated_mask == 1
        mask = self.per_new_task(self.per_task_masks[0])
        mask_list = []
        for key, param in model.named_parameters():
            if param.requires_grad and key in key_list:
                m = mask[key]
                mask_list.append(m)

        self.optimizer_g.step(mask=mask_list)
        # ================ check model update ========================
        del self.images
        del self.labels
        torch.cuda.empty_cache()

    def incremental_update(self, novel_dataset, mask=None):
        train_opt = self.opt['val']

        test_type = train_opt.get('test_type', 'NCM')

        if mask is None or self.task_id == 1:
            mask = self.per_task_masks[self.sess_id]

        if test_type == 'NCM' or self.now_session_id == 0:
            prototypes_list, labels_list = self.get_prototypes(novel_dataset, mask)
            # update prototypes dict
            for i in range(prototypes_list.shape[0]):
                self.prototypes_dict.update({labels_list[i].item(): prototypes_list[i]})

    def incremental_test(self, test_dataset, task_id=-1, test_id=-1, mask=None):
        self.net_g.eval()
        train_opt = self.opt['val']

        if mask is None or task_id == 1:
            mask = self.per_task_masks[self.sess_id]

        test_type = train_opt.get('test_type', 'NCM')
        if test_type == 'NCM' or self.now_session_id == 0:
            acc = self.__NCM_incremental_test(test_dataset, task_id, test_id, mask=mask)
        else:
            raise ValueError(f'Do not support the type {test_type} for incremental testing!')

        return acc

    def incremental_fine_tune(self, train_dataset, val_dataset, num_epoch,
                              task_id=-1, test_id=-1, tb_logger=None, mask=None):
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
                self.incremental_optimize_parameters(current_iter, mask=mask)

    def per_new_task(self, per_task_mask):
        per_new_task = deepcopy(per_task_mask)
        for key in per_task_mask.keys():
            if per_task_mask[key] is not None:
                per_new_task[key] = 1-per_task_mask[key]

        return per_new_task

    def __NCM_incremental_test(self, test_dataset, task_id=-1, test_id=-1, mask=None):
        prototypes = []
        pt_labels = []
        for key, value in self.prototypes_dict.items():
            prototypes.append(value)
            pt_labels.append(key)

        prototypes = torch.stack(prototypes).to(self.device)
        pt_labels = torch.tensor(pt_labels).to(self.device)

        if len(self.tr_labels) > 0:
            self.tr_labels = np.concatenate(self.tr_labels)

            print("==task_id:{}, test_id:{} w/ train labels{}:from {} to {}".format(task_id, test_id, len(self.tr_labels), self.tr_labels.min(), self.tr_labels.max()))
        print("==task_id:{}, test_id:{} w/ test labels{}:from {} to {}".format(task_id, test_id, prototypes.size(), pt_labels.min(), pt_labels.max()))

        p_norm = self.opt['val'].get('p_norm', None)
        if p_norm is not None and self.now_session_id > 0:
            prototypes = pnorm(prototypes, p_norm)

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
        for idx, data in enumerate(data_loader):
            self.feed_data(data)
            self.test(mask=mask)

            if self.opt.get('details', False):
                if self.labels.item() not in self.novel_classes:
                    former_prototypes = self.former_proto_list
                    logits = pair_euclidean_distances(self.output, former_prototypes)
                    estimate = torch.argmin(logits, dim=1)
                    estimate_labels = self.former_proto_label[estimate]
                    acc = (estimate_labels ==
                           self.labels).sum() / float(estimate_labels.shape[0])
                    acc_former_ave.add(acc.item(), int(estimate_labels.shape[0]))

                    logits = pair_euclidean_distances(self.output, prototypes)
                    estimate = torch.argmin(logits, dim=1)
                    estimate_labels = pt_labels[estimate]
                    acc = (estimate_labels ==
                           self.labels).sum() / float(estimate_labels.shape[0])
                    acc_former_all_ave.add(acc.item(), int(estimate_labels.shape[0]))
                    print("==test data:{} w/ labels:{} ".format(self.labels.size(),
                                                                self.labels))

                else:
                    logits = pair_euclidean_distances(self.output, prototypes)
                    estimate = torch.argmin(logits, dim=1)
                    estimate_labels = pt_labels[estimate]
                    acc = (estimate_labels ==
                           self.labels).sum() / float(estimate_labels.shape[0])
                    acc_novel_all_ave.add(acc.item(), int(estimate_labels.shape[0]))

            if self.use_cosine:
                pairwise_distance = pair_norm_cosine_distances(self.output, prototypes)
            else:
                pairwise_distance = pair_euclidean_distances(self.output, prototypes)

            estimate = torch.argmin(pairwise_distance, dim=1)
            estimate_labels = pt_labels[estimate]
            acc = (estimate_labels ==
                   self.labels).sum() / float(estimate_labels.shape[0])

            acc_ave.add(acc.item(), int(estimate_labels.shape[0]))

        # tentative for out of GPU memory
        del self.images
        del self.labels
        del self.output
        torch.cuda.empty_cache()

        log_str = f'[Test_acc of task {task_id} on test {test_id}: {acc_ave.item():.5f}]'
        logger = get_root_logger()
        logger.info(log_str)

        return acc_ave.item()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params,
                                                **train_opt['optim_g'])
        elif optim_type == 'SGD':
            self.optimizer_g = torch.optim.SGD(optim_params,
                                               **train_opt['optim_g'])
        elif optim_type == 'iSGD':
            self.optimizer_g = SGD(optim_params,
                                   **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)
        train_opt['optim_g']['type'] = optim_type

    def get_masks(self, sess_id):
        per_task_mask = self.net_g.get_masks(sess_id)
        return per_task_mask

    def feed_data(self, data):
        """
        The Data structure is (images, labels, labels_softmax)
        """
        self.images = data[0].to(self.device)
        self.labels = data[1].to(self.device)
        self.labels_softmax = data[2].to(self.device)

    def test(self, mask=None, mode='test'):
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(self.images, mask=mask, mode=mode)

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
            features_list.append(self.output)
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

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, name=''):
        logger = get_root_logger()
        logger.info('Only support single GPU validation.')
        acc = self.nondist_validation(dataloader, current_iter, tb_logger,
                                      save_img, name)
        return acc

    def nondist_validation(self, training_dataset, dataloader, current_iter,
                           tb_logger, name='', mask=None):
        """
        Args:
            current_iter: the current iteration. -1 means the testing procedure
        """
        self.net_g.eval()

        if mask is None :
            mask = self.per_task_masks[self.sess_id]

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
