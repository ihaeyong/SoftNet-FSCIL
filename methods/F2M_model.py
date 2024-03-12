import importlib
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from collections import OrderedDict
from copy import deepcopy
import wandb
import numpy as np
from os import path as osp

from methods import networks as networks
from methods.base_model import BaseModel
from utils import ProgressBar, get_root_logger, Averager, AvgDict
from metrics import pair_euclidean_distances, norm_cosine_distances


loss_module = importlib.import_module('methods.losses')


class F2MModel(BaseModel):
    """Metric-based learning model"""

    def __init__(self, opt):
        super(F2MModel, self).__init__(opt)
        self.use_cosine = self.opt.get('use_cosine', False)

        # define network
        self.net_g = networks.define_net_g(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_model_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path']['strict_load'])

        if self.opt['train'].get('fix_backbone', False):
            for k, v in self.net_g.named_parameters():
                if k.find('fc') == -1 and k.find('classifier') == -1:
                    v.requires_grad = False
                else:
                    if self.opt['train']['reset_fc'] and 'weight' in k:
                        init.normal(v, std=0.001)
                    if self.opt['train']['reset_fc'] and 'bias' in k:
                        init.constant_(v, 0)

        # generate random noise sampler
        self.generate_random_samplers(self.net_g)

        if self.is_train or self.is_incremental:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        # define losses
        self.loss_func = nn.CrossEntropyLoss()

        if train_opt.get('proto_loss'):
            proto_loss_type = train_opt['proto_loss'].pop('type')
            proto_loss_func = getattr(loss_module, proto_loss_type)
            self.proto_loss_func = proto_loss_func(**train_opt['proto_loss']).cuda()
        else:
            self.proto_loss_func = None

        # TODO define another loss (regularization), using self.name to save the loss function

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def incremental_init(self, train_set, val_set):
        """ Initializing the incremental learning procedure
        Args:
            train_set (torch.utils.data.Dataset): the training dataset
            val_set (torch.utils.data.Dataset): the validation dataset
        """

    def setup_optimizers(self):
        train_opt = self.opt['train']
        lr_cf = train_opt['optim_g'].get('lr_cf', None)

        if lr_cf is not None:
            train_opt['optim_g'].pop('lr_cf')
            opitm_embed = []
            optim_cf = []
            for k, v in self.net_g.named_parameters():
                if v.requires_grad:
                    if 'classifier' in k:
                        optim_cf.append(v)
                    else:
                        opitm_embed.append(v)
                else:
                    logger = get_root_logger()
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

    def feed_buffer_data(self, buffer_data):
        self.buffer_images = buffer_data[0]
        self.buffer_labels = buffer_data[1]
        self.buffer_labels_softmax = buffer_data[2]
        index = []
        data_labels = self.labels.unique_consecutive().cpu().numpy().tolist()
        for i, label in enumerate(self.buffer_labels):
            if label in data_labels:
                index.append(i)

        self.buffer_labels = self.buffer_labels[index].cuda()
        self.buffer_images = self.buffer_images[index].cuda()
        self.buffer_labels_softmax = self.buffer_labels_softmax[index].cuda()


    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        original_embedding, original_output = self.net_g.forward_o_embeddings(self.images)

        l_total = 0

        self.log_dict = AvgDict()
        if self.random_noise is not None:
            origin_params = [param.clone() for param in self.range_params]
            if 'Random' in self.opt['train']['random_noise']['distribution']['type']:
                self.samplers, self.bound_value = self._generate_random_samplers \
                    (self.random_noise, self.range_params)
            l_total_list = []
            for _ in range(self.random_times):
                # add noise to params
                for i in range(len(self.range_params)):
                    p = self.range_params[i]
                    sampler = self.samplers[i]
                    noise = sampler.sample()

                    p.data = p.data + noise

                noise_embedding, noise_output = self.net_g.forward_o_embeddings(self.images)

                loss1 = self.loss_func(noise_output, self.labels_softmax)
                l_total += loss1
                log_dict = {'CELoss': loss1.item()}
                self.log_dict.add_dict(log_dict)

                if self.proto_loss_func is not None:
                    proto_loss, log = self.proto_loss_func(original_embedding, noise_embedding)
                    l_total += proto_loss
                    self.log_dict.add_dict(log)

                l_total_list.append(loss1+proto_loss)
                # reset the params
                for i in range(len(self.range_params)):
                    p = self.range_params[i]
                    p.data = origin_params[i].data

            if self.opt.get('current_point', False):
                loss = self.loss_func(original_output, self.labels_softmax)
                l_total_list.append(loss)
                l_total += loss
                l_total = l_total / (self.random_times + 1)
            else:
                l_total = l_total / self.random_times

            #
            l_total_list = torch.stack(l_total_list)
            std = torch.std(l_total_list)
            l_total += 2.0 * std
            #

        else:
            loss = self.loss_func(original_output, self.labels_softmax)
            log_dict = {'CELoss': loss.item()}
            self.log_dict.add_dict(log_dict)
            l_total += loss


        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.log_dict.get_ordinary_dict()

    def test(self, output_embedding=False):
        self.net_g.eval()
        with torch.no_grad():
            if output_embedding:
                self.embeddings, self.output = self.net_g.forward_o_embeddings(self.images)
            else:
                self.output = self.net_g(self.images)
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, name=''):
        logger = get_root_logger()
        logger.info('Only support single GPU validation.')
        acc = self.nondist_validation(dataloader, current_iter, tb_logger, save_img, name)
        return acc

    def nondist_validation(self, training_dataset, dataloader, current_iter, tb_logger, name=''):
        """
        Args:
            current_iter: the current iteration. -1 means the testing procedure
        """
        self.net_g.eval()
        acc = self.__nondist_validation(training_dataset, dataloader)
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

        if self.random_noise_val is not None:
            acc_noise_avg = Averager()
            origin_params = [param.clone() for param in self.range_params_val]

            if 'Random' in self.opt['train']['random_noise']['distribution']['type']:
                self.samplers, self.bound_value = self._generate_random_samplers \
                    (self.random_noise, self.range_params)

            for _ in range(self.random_times_val):
                # add noise to params
                with torch.no_grad():
                    for i in range(len(self.range_params_val)):
                        p = self.range_params_val[i]
                        sampler = self.samplers_val[i]
                        noise = sampler.sample().cuda()
                        p.data = p.data + noise

                acc_noise = self.__nondist_validation(training_dataset, dataloader)
                acc_noise_avg.add(acc_noise)

                log_str = f'Val_acc_{_} \t {acc_noise:.5f}\n'
                logger = get_root_logger()
                logger.info(log_str)

                if current_iter != -1:
                    if tb_logger:
                        tb_logger.add_scalar(f'Val_acc_{_}', acc_noise, current_iter)
                    if self.wandb_logger is not None:
                        self.wandb_logger.log({f'Val_acc_{_}': acc_noise}, step=current_iter)
                else:
                    if tb_logger:
                        tb_logger.add_scalar(f'val_acc', acc_noise, _+1)
                    if self.wandb_logger is not None:
                        self.wandb_logger.log({f'val_acc': acc_noise}, step=_+1)

                # reset the params
                with torch.no_grad():
                    for i in range(len(self.range_params_val)):
                        p = self.range_params_val[i]
                        p.data = origin_params[i].data

        return acc

    def __nondist_validation(self, training_dataset, dataloader):
        acc_ave = Averager()

        for idx, data in enumerate(dataloader):
            self.feed_data(data)
            self.test(output_embedding=False)

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

    def test_flatness(self, dataloader, minima_acc_avg, minima_loss_avg):
        loss_ave = Averager()
        acc_ave = Averager()
        loss_func = nn.CrossEntropyLoss()

        if len(minima_acc_avg) == 0 and len(minima_loss_avg) == 0:
            for idx, data in enumerate(dataloader):
                self.feed_data(data)
                self.test(output_embedding=False)

                loss = loss_func(self.output, self.labels_softmax)
                minima_loss_avg.add(loss.item(), int(self.labels_softmax.shape[0]))

                estimate_labels = torch.argmax(self.output, dim=1)
                acc = (estimate_labels ==
                       self.labels_softmax).sum() / float(estimate_labels.shape[0])

                minima_acc_avg.add(acc.item(), int(self.labels_softmax.shape[0]))

        if self.random_noise is not None:
            origin_params = [param.clone() for param in self.range_params]
            if 'Random' in self.opt['train']['random_noise']['distribution']['type']:
                self.samplers, self.bound_value = self._generate_random_samplers \
                    (self.random_noise, self.range_params)
            for i in range(len(self.range_params)):
                p = self.range_params[i]
                sampler = self.samplers[i]
                noise = sampler.sample()
                p.data = p.data + noise

            for idx, data in enumerate(dataloader):
                self.feed_data(data)
                self.test(output_embedding=False)

                loss = loss_func(self.output, self.labels_softmax)
                loss_ave.add(loss.item(), int(self.labels_softmax.shape[0]))

                estimate_labels = torch.argmax(self.output, dim=1)
                acc = (estimate_labels ==
                       self.labels_softmax).sum() / float(estimate_labels.shape[0])

                acc_ave.add(acc.item(), int(self.labels_softmax.shape[0]))

            # reset the params
            for i in range(len(self.range_params)):
                p = self.range_params[i]
                p.data = origin_params[i].data

            return loss_ave.item(), acc_ave.item()

        else:
            for idx, data in enumerate(dataloader):
                self.feed_data(data)
                self.test(output_embedding=False)

                loss = loss_func(self.output, self.labels_softmax)
                loss_ave.add(loss.item(), int(self.labels_softmax.shape[0]))

                estimate_labels = torch.argmax(self.output, dim=1)
                acc = (estimate_labels ==
                       self.labels_softmax).sum() / float(estimate_labels.shape[0])

                acc_ave.add(acc.item(), int(self.labels_softmax.shape[0]))
            return loss_ave.item(), acc_ave.item()

    def test_flatness2(self, dataloader):
        loss_ave = Averager()
        acc_ave = Averager()
        loss_func = nn.CrossEntropyLoss()

        for idx, data in enumerate(dataloader):
            if self.random_noise is not None:
                origin_params = [param.clone() for param in self.range_params]
                if 'Random' in self.opt['train']['random_noise']['distribution']['type']:
                    self.samplers, self.bound_value = self._generate_random_samplers \
                        (self.random_noise, self.range_params)
                for _ in range(self.random_times):
                    # add noise to params
                    for i in range(len(self.range_params)):
                        p = self.range_params[i]
                        sampler = self.samplers[i]
                        noise = sampler.sample()

                        p.data = p.data + noise

                self.feed_data(data)
                self.test(output_embedding=False)

                loss = loss_func(self.output, self.labels_softmax)
                loss_ave.add(loss.item(), int(self.labels_softmax.shape[0]))

                estimate_labels = torch.argmax(self.output, dim=1)
                acc = (estimate_labels ==
                       self.labels_softmax).sum() / float(estimate_labels.shape[0])

                acc_ave.add(acc.item(), int(self.labels_softmax.shape[0]))

                # reset the params
                for i in range(len(self.range_params)):
                    p = self.range_params[i]
                    p.data = origin_params[i].data
            else:
                for idx, data in enumerate(dataloader):
                    self.feed_data(data)
                    self.test(output_embedding=False)

                    loss = loss_func(self.output, self.labels_softmax)
                    loss_ave.add(loss.item(), int(self.labels_softmax.shape[0]))

                    estimate_labels = torch.argmax(self.output, dim=1)
                    acc = (estimate_labels ==
                           self.labels_softmax).sum() / float(estimate_labels.shape[0])

                    acc_ave.add(acc.item(), int(self.labels_softmax.shape[0]))

        loss_list = loss_ave.obtain_data()
        loss_m = np.mean(loss_list)
        loss_std = np.std(loss_list)

        acc_list = acc_ave.obtain_data()
        acc_m = np.mean(acc_list)
        acc_std = np.std(acc_list)

        return loss_m, loss_std, acc_m, acc_std

    def save(self, epoch, current_iter, name='net_g', dataset=None):
        self.save_network(self.net_g, name, current_iter)
        self.save_training_state(epoch, current_iter)

    def _get_features(self, dataset):
        aug = dataset.get_aug()
        dataset.set_aug(False)

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=128, shuffle=False, drop_last=False)

        features = []
        labels = []

        for i, data in enumerate(data_loader, 0):
            with torch.no_grad():
                self.feed_data(data)
                self.net_g.eval()
                output = self.net_g.forward_without_cf(self.images)
                features.append(output)
                labels.append(self.labels)

        del self.images
        del self.labels
        torch.cuda.empty_cache()

        dataset.set_aug(aug)

        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)
        return features, labels

    # def obtain_buffer_index(self, dataset, buffer_size=5):
    #     buffer_approach = self.opt.get('buffer_approach', 'normal')
    #     if buffer_approach == 'normal':
    #         buffer_indices = self.__normal_obtain_buffer_index(dataset, buffer_size)
    #     elif buffer_approach == 'greedy':
    #         buffer_indices = self.__greedy_obtain_buffer_index(dataset, buffer_size)
    #     else:
    #         raise ValueError(f'Do not define the approach of obtaining buffer: {buffer_approach}')
    #
    #     return buffer_indices

    def get_prototypes(self, dataset):
        """
        calculated the prototypes for each class in training dataset

        Args:
            dataset (torch.utils.data.Dataset): the dataset

        Returns:
            tuple: (prototypes_list, labels_list) where prototypes_list is the list of prototypes and
            labels_list is the list of class labels
        """
        features, labels = self._get_features(dataset)
        selected_classes = dataset.selected_classes

        prototypes_list = []

        for cl in selected_classes:
            index_cl = torch.where(cl == labels)[0]
            class_features = features[index_cl]
            prototypes_list.append(class_features.mean(dim=0))

        prototypes_list = torch.stack(prototypes_list, dim=0)

        return prototypes_list, torch.from_numpy(dataset.selected_classes)

    def feed_prototypes(self, prototypes, labels):
        self.prototypes = prototypes.cuda()
        self.pt_labels = labels.cuda()
        self.pt_labels_dict = {pt_label.item(): i  for i, pt_label in enumerate(self.pt_labels)}

    # def __normal_obtain_buffer_index(self, dataset, buffer_size):
    #     features, labels = self._get_features(dataset)
    #     class_labels = torch.unique_consecutive(labels)
    #
    #     prototypes = []
    #     for cl in class_labels:
    #         index_cl = torch.where(labels==cl.item())[0]
    #         class_features = features[index_cl]
    #         prototypes.append(class_features.mean(dim=0))
    #
    #     prototypes = torch.stack(prototypes, dim=0)
    #
    #     buffer_indices = []
    #     for i, class_label in enumerate(class_labels):
    #         feat_index = (labels == class_label.item()).nonzero().squeeze()
    #         feats = features[feat_index]
    #         proto = prototypes[i]
    #         proto = proto.unsqueeze(dim=0).expand(feats.shape[0], -1)
    #         logits = - ((feats - proto) ** 2).sum(dim=1)
    #         values, indices = torch.topk(input=logits, k=buffer_size)
    #         indices = feat_index[indices]
    #         buffer_indices.append(indices)
    #
    #     buffer_indices = torch.stack(buffer_indices).reshape(-1)
    #
    #     return buffer_indices.cpu().numpy().tolist()
    #
    # def __greedy_obtain_buffer_index(self, dataset, buffer_size):
    #     features, labels = self._get_features(dataset)
    #     if self.use_cosine:
    #         features = F.normalize(features, dim=1)
    #         #norm = features.norm(dim=1)
    #
    #     class_labels = torch.unique_consecutive(labels)
    #
    #     prototypes = []
    #     for cl in class_labels:
    #         index_cl = torch.where(labels == cl.item())[0]
    #         class_features = features[index_cl]
    #         prototypes.append(class_features.mean(dim=0))
    #
    #     prototypes = torch.stack(prototypes, dim=0)
    #
    #     buffer_indices = []
    #     buffer_indices_per_class = []
    #
    #     for i, class_label in enumerate(class_labels):
    #         prototype = prototypes[i]
    #         for idx in range(self.opt['train']['buffer_size']):
    #             feat_index = (labels == class_label.item()).nonzero().squeeze()
    #             feat_index_exclude = [index.item() for index in feat_index if
    #                                   index.item() not in buffer_indices_per_class]
    #
    #             if len(buffer_indices_per_class) > 0:
    #                 class_feats_include = features[buffer_indices_per_class]
    #                 class_proto_include = class_feats_include.mean(dim=0)
    #                 class_feats_exclude = features[feat_index_exclude]
    #                 fake_prototypes = 1.0 / (idx + 1.0) * class_feats_exclude + float(idx) / (
    #                             idx + 1.0) * class_proto_include
    #
    #                 prototype_expand = prototype.unsqueeze(dim=0).expand(len(fake_prototypes), -1)
    #
    #                 if self.use_cosine:
    #                     logits = norm_cosine_distances(prototype_expand, fake_prototypes)
    #                 else:
    #                     logits = ((prototype_expand - fake_prototypes) ** 2).mean(dim=1)
    #
    #                 min_v, _ = torch.min(logits, dim=0)
    #                 min_index = torch.argmin(logits, dim=0)
    #                 buffer_indices_per_class.append(feat_index_exclude[min_index])
    #
    #             else:
    #                 class_feats_exclude = features[feat_index_exclude]
    #                 prototype_expand = prototype.unsqueeze(dim=0).expand(len(class_feats_exclude), -1)
    #
    #                 if self.use_cosine:
    #                     logits = norm_cosine_distances(prototype_expand, class_feats_exclude)
    #                 else:
    #                     logits = ((prototype_expand - class_feats_exclude) ** 2).mean(dim=1)
    #
    #                 min_index = torch.argmin(logits, dim=0)
    #                 buffer_indices_per_class.append(feat_index_exclude[min_index])
    #
    #         buffer_indices = buffer_indices + buffer_indices_per_class
    #         buffer_indices_per_class = []
    #
    #     return buffer_indices

    def __get_centers(self, dataset):
        prototypes, labels = self.get_prototypes(dataset)
        centers = nn.Parameter(torch.randn(self.opt['train']['bases'], self.opt['network_g']['Embed_dim']).cuda())
        return centers, labels.cuda()
