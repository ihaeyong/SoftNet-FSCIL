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
from utils import ProgressBar, get_root_logger, Averager, dir_size, AvgDict, pnorm
from data.normal_dataset import NormalDataset
from data import create_sampler, create_dataloader, create_dataset
from metrics import pair_euclidean_distances, pair_euclidean_distances_dim3
from metrics.norm_cosine_distances import pair_norm_cosine_distances, pair_norm_cosine_distances_dim3

loss_module = importlib.import_module('methods.losses')

class MTNBModel(BaseModel):
    """Metric-based with random noise to parameters learning model"""
    def __init__(self, opt):
        super(MTNBModel, self).__init__(opt)

        self.use_cosine = self.opt.get('use_cosine', False)
        # the index of buffer data
        self.sample_index = None

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
            self.net_g_first = deepcopy(self.net_g)
            self.net_g_first.eval()

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

        self.generate_random_samplers(self.net_g)
        if self.random_noise is not None:
            self.net_g_first_params = self._obtain_embedding_params(self.range_params_name, self.net_g_first)

        if self.is_train or (self.is_incremental and self.opt['train']['fine_tune']):
             self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('pn_opt'):
            metric_type = train_opt['pn_opt'].pop('type')
            pn_loss_func = getattr(loss_module, metric_type)
            self.pn_loss_func = pn_loss_func(**train_opt['pn_opt']).cuda()
        else:
            self.pn_loss_func = None

        # regularization
        self.img_buffer = None

        # fix the parameters of backbone exclude range parameters
        if self.is_incremental and self.now_session_id > 0 and self.random_noise is not None:
            for k, v in self.net_g.named_parameters():
                if k not in self.range_params_name:
                    v.requires_grad = False

        # save the range params as original parameters
        if self.is_incremental and self.now_session_id > 0 and self.random_noise is not None:
            self.save_original_range_params()

        self.setup_optimizers()
        self.setup_schedulers()

    def train_buffer_embeddings(self):
        self.net_g.train()
        buffer_embeddings = self.net_g(self.buffer_images.cuda())
        return buffer_embeddings

    def incremental_optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.net_g.eval()

        output = self.net_g(self.images)

        l_total = 0.0
        self.log_dict = AvgDict()

        if self.pn_loss_func is not None:
            loss, log = self.pn_loss_func(self.former_proto_list, self.former_proto_label, output,
                                          self.labels)
            l_total += loss
            self.log_dict.add_dict(log)

        l_total.backward()
        n_steps = 1
        max_steps = self.opt['train']['max_steps']
        ratio = self.opt['train']['ratio']
        if self.opt['train'].get('rounding', True):
            for p in self.range_params:
                factor = self.random_noise['bound_value'] / self.random_noise['reduction_factor']
                lr = self.get_current_learning_rate()[0]
                grad = p.grad * lr
                grad = grad / factor
                grad = torch.round(grad)
                if (grad == 0).sum() == torch.numel(p):
                    while n_steps <= max_steps and (grad == 0).sum() == torch.numel(p):
                        n_steps += 1
                        grad = p.grad * lr * n_steps
                        grad = grad / factor
                        grad = torch.round(grad)
                    n_steps = 1

                grad = grad * factor / (lr * ratio)
                grad_mean = grad.mean()
                p.grad = grad


        self.optimizer_g.step()

        if self.opt['train'].get('noise', True):
            for i, p in enumerate(self.range_params):
                p2 = self.net_g_first_params[i]
                diff = p - p2
                factor = self.random_noise['bound_value'] / self.random_noise['reduction_factor']
                change = torch.round(diff / factor)

        self.log_dict = self.log_dict.get_ordinary_dict()


    def incremental_init(self, train_set, val_set):
        """ Initializing the incremental learning procedure
        Args:
            train_set (torch.utils.data.Dataset): the training dataset
            val_set (torch.utils.data.Dataset): the validation dataset
        """
        self.novel_classes = train_set.selected_classes

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
            if self.opt.get('details', False):
                acc, acc_former_ave, acc_former_all_ave, acc_novel_all_ave = self.__NCM_incremental_test(test_dataset, task_id, test_id)
            else:
                acc = self.__NCM_incremental_test(test_dataset, task_id, test_id)
        else:
            raise ValueError(f'Do not support the type {test_type} for testing')

        if self.opt.get('details', False):
            return acc, acc_former_ave, acc_former_all_ave, acc_novel_all_ave
        else:
            return acc

    def __NCM_incremental_test(self, test_dataset, task_id=-1, test_id=-1):
        prototypes = []
        pt_labels = []
        for key, value in self.prototypes_dict.items():
            prototypes.append(value)
            pt_labels.append(key)

        prototypes = torch.stack(prototypes).cuda()
        pt_labels = torch.tensor(pt_labels).cuda()

        # p_norm = self.opt['val'].get('p_norm', None)
        # if p_norm is not None and self.now_session_id > 0:
        p_norm = 1.0
        prototypes = pnorm(prototypes, p_norm)

        if self.opt.get('details', False):
            data_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, drop_last=False)
        else:
            data_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=False,
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
            self.test()
            if self.opt.get('details', False):
                if self.labels.item() not in self.novel_classes:
                    if p_norm is not None and self.now_session_id > 0:
                        former_prototypes = pnorm(self.former_proto_list, p_norm)
                    else:
                        former_prototypes = self.former_proto_list
                    if self.use_cosine:
                        logits = pair_norm_cosine_distances(self.output, former_prototypes)
                    else:
                        logits = pair_euclidean_distances(self.output, former_prototypes)
                    estimate = torch.argmin(logits, dim=1)
                    estimate_labels = self.former_proto_label[estimate]
                    acc = (estimate_labels ==
                           self.labels).sum() / float(estimate_labels.shape[0])
                    acc_former_ave.add(acc.item(), int(estimate_labels.shape[0]))

                    if self.use_cosine:
                        logits = pair_norm_cosine_distances(self.output, prototypes)
                    else:
                        logits = pair_euclidean_distances(self.output, prototypes)
                    estimate = torch.argmin(logits, dim=1)
                    estimate_labels = pt_labels[estimate]
                    acc = (estimate_labels ==
                           self.labels).sum() / float(estimate_labels.shape[0])
                    acc_former_all_ave.add(acc.item(), int(estimate_labels.shape[0]))

                else:
                    if self.use_cosine:
                        logits = pair_norm_cosine_distances(self.output, prototypes)
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

            # # tentative for out of GPU memory
            # del self.images
            # del self.labels
            # del self.output
            # torch.cuda.empty_cache()
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
        train_dataset_temp = train_dataset

        sampler_opt = self.opt['datasets']['train']['sampler']
        sampler_opt['num_classes'] = self.num_novel_class

        train_sampler = create_sampler(train_dataset_temp, sampler_opt)
        dataset_opt = self.opt['datasets']['train']

        train_loader = create_dataloader(
            train_dataset_temp,
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
                if self.random_noise is not None:
                    if self.opt['train'].get('clamping', True):
                        self.clamp_range_params()

                logger = get_root_logger()
                message = f'[epoch:{epoch:3d}, iter:{current_iter:4,d}, lr:({self.get_current_learning_rate()[0]:.3e})] [ '

                for key, value in self.log_dict.items():
                    message += f'{key}: {value:.4f}, '

                logger.info(message + ']')

                if self.opt['val']['val_freq'] is not None and current_iter % self.opt[
                    'val']['val_freq'] == 0:
                    self.incremental_update(novel_dataset=train_dataset_temp)

                    log_str = f'Epoch {epoch}, Validation Step:\n'
                    logger = get_root_logger()
                    logger.info(log_str)
                    acc = self.incremental_test(val_dataset, task_id=task_id, test_id=test_id)
                    if tb_logger:
                        tb_logger.add_scalar(f'val_acc_of_session{task_id}_test{test_id}', acc, current_iter)
                    if self.wandb_logger is not None:
                        wandb.log({f'val_acc_of_session{task_id}_test{test_id}': acc, f'ft_step': current_iter})

    def setup_schedulers_params(self):
        train_opt = self.opt['train']
        if self.is_incremental and train_opt['fine_tune']:
            train_opt = self.opt['train']
            sampler_opt = self.opt['datasets']['train']['sampler']
            if train_opt['buffer_size'] > 0:
                total_images = self.total_class * sampler_opt['num_samples']
            else:
                total_images = self.num_novel_class * sampler_opt['num_samples']

            batch_size = train_opt['fine_tune_batch']

            iteration_per_epoch = int(total_images / batch_size)

            for key, value in train_opt['scheduler'].items():
                if isinstance(value, list):
                    train_opt['scheduler'][key] = [iteration_per_epoch * epoch for epoch in value]

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

    def feed_novel_data(self, data):
        self.novel_images = data[0].cuda()
        self.novel_labels = data[1].cuda()
        self.novel_labels_softmax = data[2].cuda()

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(self.images)

    def test_former_embedding(self, images, norm=False):
        self.net_g_former.eval()
        with torch.no_grad():
            buffer_output = self.net_g_former(images)
            if norm:
                buffer_output = F.normalize(buffer_output, dim=1)
        return buffer_output

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

    def save(self, epoch, current_iter, name='net_g', dataset=None):
        self.save_network(self.net_g, name, current_iter)
        self.save_training_state(epoch, current_iter)
        if self.is_incremental:
            self.save_prototypes(self.now_session_id, self.opt['test_id'])

    def obtain_buffer_index(self, dataset):
        if self.now_session_id == 0 and self.opt['path'].get('base_buffer') is not None:
            path = self.opt['path'].get('base_buffer')
            index = torch.load(path).numpy().tolist()
            self.sample_index = index
            return

        features, label = self._get_features(dataset)
        class_labels = torch.unique_consecutive(label)

        sample_index = []
        sample_index_per_class = []

        for class_label in class_labels:
            prototype = self.prototypes_dict[class_label.item()].cpu()
            for idx in range(self.opt['train']['buffer_size']):
                feat_index = (label == class_label.item()).nonzero().squeeze()
                feat_index_exclude = [index.item() for index in feat_index if index.item() not in sample_index_per_class]

                if len(sample_index_per_class) > 0:
                    class_feats_include = features[sample_index_per_class]
                    class_proto_include = class_feats_include.mean(dim=0)
                    class_feats_exclude = features[feat_index_exclude]
                    fake_prototypes = 1.0 / (idx + 1.0) * class_feats_exclude + float(idx) / (idx + 1.0) * class_proto_include

                    prototype_expand = prototype.unsqueeze(dim=0).expand(len(fake_prototypes), -1)

                    logits = ((prototype_expand - fake_prototypes) ** 2).mean(dim=1)

                    min_index = torch.argmin(logits, dim=0)
                    sample_index_per_class.append(feat_index_exclude[min_index])

                else:
                    class_feats_exclude = features[feat_index_exclude]
                    prototype_expand = prototype.unsqueeze(dim=0).expand(len(class_feats_exclude), -1)
                    logits = ((prototype_expand - class_feats_exclude) ** 2).mean(dim=1)

                    min_index = torch.argmin(logits, dim=0)
                    sample_index_per_class.append(feat_index_exclude[min_index])

            sample_index = sample_index + sample_index_per_class
            sample_index_per_class = []

        self.sample_index = sample_index

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
            prototypes_dict = torch.load(load_path)
            self.prototypes_dict = prototypes_dict
            self.former_proto_list, self.former_proto_label = self._read_prototypes()
        else:
            if self.opt['path'].get('pretrain_prototypes', None) is not None:
                prototypes_dict = torch.load(self.opt['path']['pretrain_prototypes'])
                self.prototypes_dict = prototypes_dict
                self.former_proto_list, self.former_proto_label = self._read_prototypes()

    def set_the_saving_files_path(self, opt, task_id, test_id):
        # change the path of base model
        save_filename_g = f'test{test_id}_session_{task_id}.pth'
        save_path_g = osp.join(opt['path']['models'], save_filename_g)
        opt['path']['base_model'] = save_path_g

    def _get_features(self, dataset):
        aug = dataset.get_aug()
        dataset.set_aug(False)

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=16, shuffle=False, drop_last=False)

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

    def _obtain_embedding_params(self, names, net):
        params = []
        for k, value in net.named_parameters():
            if k in names:
                params.append(value)
        return params