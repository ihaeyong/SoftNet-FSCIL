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

loss_module = importlib.import_module('methods.losses')

class FSLLModel(BaseModel):
    """Metric-based with random noise to parameters learning model"""
    def __init__(self, opt):
        super(FSLLModel, self).__init__(opt)

        self.use_cosine = self.opt.get('use_cosine', False)

        if self.is_incremental:
            train_opt = self.opt['train']
            self.now_session_id = self.opt['task_id'] + 1
            self.num_novel_class = train_opt['num_class_per_task'] if self.now_session_id > 0 else 0
            self.total_class = train_opt['bases'] + self.num_novel_class * self.now_session_id
            self.num_old_class = self.total_class - self.num_novel_class if self.now_session_id > 0 else self.total_class

        # define network
        self.net_g = networks.define_net_g(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        if not (self.is_incremental and self.now_session_id >0):
            self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_model_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path']['strict_load'])

        # load base models for incremental learning
        load_base_model_path = self.opt['path'].get('base_model', None)
        if load_base_model_path is not None and self.is_incremental:
            self.load_network(self.net_g, load_base_model_path,
                              self.opt['path']['strict_load'])
            # load the prototypes for all seen classes
            self.load_prototypes(opt['task_id'], opt['test_id'])

            # record the former network
            self.net_g_former = deepcopy(self.net_g)
            self.net_g_former.eval()

        if self.is_train or (self.is_incremental and self.opt['train']['fine_tune']):
             self.init_training_settings()

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
        if train_opt.get('metric_opt'):
            metric_type = train_opt['metric_opt'].pop('type')
            metric_loss_func = getattr(loss_module, metric_type)
            self.metric_loss_func = metric_loss_func(**train_opt['metric_opt']).cuda()
        else:
            self.metric_loss_func = None

        if train_opt.get('pn_opt'):
            cf_type = train_opt['pn_opt'].pop('type')
            pn_loss_func = getattr(loss_module, cf_type)
            self.pn_loss_func = pn_loss_func(**train_opt['pn_opt']).cuda()
        else:
            self.pn_loss_func = None

        # regularization
        if train_opt.get('regularization'):
            regular_type = train_opt['regularization'].pop('type')
            regularization_func = getattr(loss_module, regular_type)
            self.regularization_func = regularization_func(**train_opt['regularization']).cuda()
        else:
            self.regularization_func = None

        # # define the buffer
        # if self.is_incremental and self.now_session_id > 0 and self.opt['train']['feat_buffer']:
        #     self.img_buffer = self._load_img_buffer()
        # else:
        #     self.img_buffer = None

        self.freeze_networks_with_threshold(train_opt['threshold'])

        self.setup_optimizers()
        self.setup_schedulers()

    def incremental_optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.net_g.train()

        output = self.net_g(self.images)

        l_total = 0

        if self.metric_loss_func:
            loss, log = self.metric_loss_func(output, self.labels)
            l_total += loss
            # update the log_dict
            self.log_dict.update(log)

        if self.pn_loss_func is not None:
            loss, log = self.pn_loss_func(self.former_proto_list, self.former_proto_label, output, self.labels)
            l_total += loss
            self.log_dict.update(log)

        if self.regularization_func is not None:
            loss, log = self.regularization_func(self.former_optim_param, self.optim_param)
            l_total += loss
            self.log_dict.update(log)

        l_total.backward()
        self.optimizer_g.step()

        del self.images
        del self.labels
        torch.cuda.empty_cache()

    def incremental_update(self, novel_dataset):
        train_opt = self.opt['val']

        test_type = train_opt.get('test_type', 'NCM')

        if test_type == 'NCM' or self.now_session_id == 0:
            prototypes_list, labels_list = self.get_prototypes(novel_dataset)
            # update prototypes dict
            for i in range(prototypes_list.shape[0]):
                self.prototypes_dict.update({labels_list[i].item(): prototypes_list[i]})


    def incremental_test(self, test_dataset, task_id=-1, test_id=-1):
        self.net_g.eval()
        train_opt = self.opt['val']

        test_type = train_opt.get('test_type', 'NCM')
        if test_type == 'NCM' or self.now_session_id == 0:
            acc = self.__NCM_incremental_test(test_dataset, task_id, test_id)
        else:
            raise ValueError(f'Do not support the type {test_type} for incremental testing!')

        return acc

    def __NCM_incremental_test(self, test_dataset, task_id=-1, test_id=-1):
        prototypes = []
        pt_labels = []
        for key, value in self.prototypes_dict.items():
            prototypes.append(value)
            pt_labels.append(key)

        prototypes = torch.stack(prototypes).cuda()
        pt_labels = torch.tensor(pt_labels).cuda()

        print("==prototypes:{} w/ labels:{}".format(prototypes.size(), pt_labels))

        p_norm = self.opt['val'].get('p_norm', None)
        if p_norm is not None and self.now_session_id > 0:
            prototypes = pnorm(prototypes, p_norm)

        data_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, drop_last=False)

        acc_ave = Averager()
        acc_former_ave = Averager()
        acc_former_all_ave = Averager()
        acc_novel_all_ave = Averager()

        # test norm
        novel_norm = Averager()
        old_norm = Averager()

        for idx, data in enumerate(data_loader):
            self.feed_data(data)
            self.test()

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
                    print("==test data:{} w/ labels:{}".format(self.labels.size(), self.labels))

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

    def incremental_fine_tune(self, train_dataset, val_dataset, num_epoch, task_id=-1, test_id=-1, tb_logger=None):
        """
        fine tune the models with the samples of incremental novel class

        Args:
            train_dataset (torch.utils.data.Dataset): the training dataset
            val_dataset (torch.utils.data.Dataset): the validation dataset
            num_epoch (int): the number of epoch to fine tune the models
            task_id (int): the id of sessions
            test_id (int): the id of few-shot test
        """


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
            for idx, data in enumerate(train_loader):
                current_iter += 1
                self.update_learning_rate(
                    current_iter, warmup_iter=-1)

                self.feed_data(data)
                self.incremental_optimize_parameters(current_iter)

                logger = get_root_logger()
                message = f'[epoch:{epoch:3d}, iter:{current_iter:4,d}, lr:({self.get_current_learning_rate()[0]:.3e})] [ '
                for key, value in self.log_dict.items():
                    message += f'{key}: {value:.4f}, '

                logger.info(message + ']')
                if tb_logger:
                    tb_logger.add_scalar(f'ft_loss_of_session{task_id}_test{test_id}', self.log_dict['Loss'],
                                         current_iter)
                    tb_logger.add_scalar(f'acc_of_session{task_id}_test{test_id}', self.log_dict['Tr_acc'],
                                         current_iter)
                if self.wandb_logger is not None:
                    wandb.log(
                        {f'ft_loss_of_session{task_id}_test{test_id}': self.log_dict['Loss'], f'ft_step': current_iter})
                    wandb.log(
                        {f'acc_of_session{task_id}_test{test_id}': self.log_dict['Tr_acc'], f'ft_step': current_iter})

                if self.opt['val']['val_freq'] is not None and current_iter % self.opt[
                    'val']['val_freq'] == 0:
                    self.incremental_update(train_dataset)
                    log_str = f'Epoch {epoch}, Validation Step:\n'
                    logger = get_root_logger()
                    logger.info(log_str)
                    acc = self.incremental_test(val_dataset, task_id=task_id, test_id=test_id)
                    if tb_logger:
                        tb_logger.add_scalar(f'val_acc_of_session{task_id}_test{test_id}', acc, current_iter)
                    if self.wandb_logger is not None:
                        wandb.log({f'val_acc_of_session{task_id}_test{test_id}': acc, f'ft_step': current_iter})

    # def setup_schedulers_params(self):
    #     train_opt = self.opt['train']
    #     if self.is_incremental and train_opt['fine_tune']:
    #         train_opt = self.opt['train']
    #         sampler_opt = self.opt['datasets']['train']['sampler']
    #         if train_opt['buffer_size'] > 0:
    #             total_images = self.total_class * sampler_opt['num_samples']
    #         else:
    #             total_images = self.num_novel_class * sampler_opt['num_samples']
    #
    #         batch_size = train_opt['fine_tune_batch']
    #
    #         iteration_per_epoch = int(total_images / batch_size)
    #
    #         for key, value in train_opt['scheduler'].items():
    #             if isinstance(value, list):
    #                 train_opt['scheduler'][key] = [iteration_per_epoch * epoch for epoch in value]

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
            self.optimizer_g = torch.optim.SGD(optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        """
        The Data structure is (images, labels, labels_softmax)
        """
        self.images = data[0].cuda()
        self.labels = data[1].cuda()
        self.labels_softmax = data[2].cuda()

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(self.images)
        self.net_g.train()

    # def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
    #     logger = get_root_logger()
    #     logger.info('Only support single GPU validation.')
    #     acc = self.nondist_validation(dataloader, current_iter, tb_logger, save_img)
    #     return acc
    #
    # def nondist_validation(self, training_dataset, dataloader, current_iter, tb_logger):
    #     self.net_g.eval()
    #     prototypes, pt_labels = self.get_prototypes(training_dataset)
    #     acc = self.__nondist_validation(dataloader, prototypes, pt_labels)
    #     log_str = f'Val_acc \t {acc:.5f}\n'
    #     logger = get_root_logger()
    #     logger.info(log_str)
    #
    #     if current_iter != -1:
    #         if tb_logger:
    #             tb_logger.add_scalar(f'val_acc', acc, current_iter)
    #         if self.wandb_logger is not None:
    #             self.wandb_logger.log({f'val_acc': acc}, step=current_iter)
    #     else:
    #         if tb_logger:
    #             tb_logger.add_scalar(f'val_acc', acc, 0)
    #         if self.wandb_logger is not None:
    #             self.wandb_logger.log({f'val_acc': acc}, step=0)
    #
    #     return acc
    #
    # def __nondist_validation(self, dataloader, prototypes, pt_labels):
    #     prototypes = prototypes.cuda()
    #     pt_labels = pt_labels.cuda()
    #     acc_ave = Averager()
    #     for idx, data in enumerate(dataloader):
    #         self.feed_data(data)
    #         self.test()
    #
    #         if self.use_cosine:
    #             pairwise_distance = pair_norm_cosine_distances(self.output, prototypes)
    #         else:
    #             pairwise_distance = pair_euclidean_distances(self.output, prototypes)
    #
    #         estimate = torch.argmin(pairwise_distance, dim=1)
    #
    #         estimate_labels = pt_labels[estimate]
    #
    #         acc = (estimate_labels ==
    #                self.labels).sum() / float(estimate_labels.shape[0])
    #
    #         acc_ave.add(acc.item(), int(estimate_labels.shape[0]))
    #
    #         # tentative for out of GPU memory
    #     del self.images
    #     del self.labels
    #     del self.output
    #     torch.cuda.empty_cache()
    #
    #     return acc_ave.item()

    def get_prototypes(self, training_dataset):
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
            self.test()
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

    # def _log_validation_metric_values(self, current_iter, dataset_name,
    #                                   tb_logger):
    #     log_str = f'Validation {dataset_name}\n'
    #     for metric, value in self.metric_results.items():
    #         log_str += f'\t # {metric}: {value:.4f}\n'
    #     logger = get_root_logger()
    #     logger.info(log_str)
    #     if tb_logger:
    #         for metric, value in self.metric_results.items():
    #             tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def save(self, epoch, current_iter, name='net_g', dataset=None):
        self.save_network(self.net_g, name, current_iter)
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
            prototypes = torch.stack(prototypes).cuda()
            pt_labels = torch.tensor(pt_labels).cuda()
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
