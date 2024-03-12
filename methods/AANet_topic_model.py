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

from dataloader_alice.data_utils import * # topic split

loss_module = importlib.import_module('methods.losses')

class AANetTopicModel(BaseModel):
    """Metric-based learning model"""
    def __init__(self, opt):
        super(AANetTopicModel, self).__init__(opt)
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
        self.net_g = self.model_to_device(self.net_g, opt['gpu'])
        self.print_network(self.net_g)

        # define extended network
        self.net_e = networks.define_net_g(deepcopy(opt['network_g']))
        self.net_e = self.model_to_device(self.net_g, opt['gpu'])
        self.print_network(self.net_e)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_model_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path']['strict_load'])

        # load base models for incremental learning
        load_base_model_path = self.opt['path'].get('base_model', None)
        if load_base_model_path is not None and self.is_incremental:
            print(f' ---- load:{load_base_model_path}')
            self.load_network(self.net_g, load_base_model_path,
                              self.opt['path']['strict_load'])

            # load the prototypes for all seen classes
            assert opt['task_id'] + 1 == self.now_session_id
            self.load_prototypes(opt['task_id'], opt['test_id'])

        # load extended models for incremental learning
        load_extra_model_path = self.opt['path'].get('extra_model', None)
        if load_extra_model_path is not None and self.now_session_id > 0:
            print(f' ---- load:{load_extra_model_path}')
            self.load_network(self.net_e, load_extra_model_path,
                              self.opt['path']['strict_load'])

        self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()

        # define losses
        self.loss_func = nn.CrossEntropyLoss()

        train_opt = self.opt['train']
        if train_opt.get('proto_loss'):
            proto_loss_type = train_opt['proto_loss'].pop('type')
            proto_loss_func = getattr(loss_module, proto_loss_type)
            self.proto_loss_func = proto_loss_func(**train_opt['proto_loss']).cuda()
        else:
            self.proto_loss_func = None

        if train_opt.get('cpn_opt'):
            cf_type = train_opt['cpn_opt'].pop('type')
            pn_loss_func = getattr(loss_module, cf_type)
            self.cpn_loss_func = pn_loss_func(**train_opt['cpn_opt']).to(self.device)
            train_opt['cpn_opt']['type'] = cf_type
        else:
            self.cpn_loss_func = None

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_e_optimizers()

        self.setup_schedulers()

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

    def setup_e_optimizers(self):
        train_opt = self.opt['train']
        lr_cf = train_opt['optim_e'].get('lr_cf', None)

        if lr_cf is not None:
            train_opt['optim_e'].pop('lr_cf')
            opitm_embed = []
            optim_cf = []
            for k, v in self.net_e.named_parameters():
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
            for k, v in self.net_e.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger = get_root_logger()
                    logger.warning(f'Params {k} will not be optimized.')

        self.optim_e_type = train_opt['optim_e'].pop('type')
        if self.optim_e_type == 'Adam':
            self.optimizer_e = torch.optim.Adam(optim_params,
                                                **train_opt['optim_e'])
        elif self.optim_e_type == 'SGD':
            self.optimizer_e = torch.optim.SGD(optim_params, **train_opt['optim_e'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_e)
        train_opt['optim_e']['type'] = self.optim_e_type

    def incremental_init(self, train_set, val_set):
        """ Initializing the incremental learning procedure
        Args:
            train_set (torch.utils.data.Dataset): the training dataset
            val_set (torch.utils.data.Dataset): the validation dataset
        """

        selected_classes = get_session_classes(
            session=self.now_session_id,
            opt=self.opt)

        if self.now_session_id > 0:
            novel_classes = selected_classes[-self.opt['train']['way']:]
        else:
            novel_classes = []
        self.novel_classes = novel_classes

    def incremental_update(self, novel_dataset, mask=None):
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

    def incremental_optimize_parameters(self, current_iter, mask=None):
        # set zero grads
        self.optimizer_g.zero_grad()
        self.optimizer_e.zero_grad()

        model = getattr(self.net_g, 'func')
        model_e = getattr(self.net_e, 'func')
        model.train()
        model_e.train()

        # ======================AANet===========================
        output_b_task, output_b_fc = model.forward_score(self.images)
        output_e_task, output_e_fc = model_e.forward_score(self.images)

        self.scale_output(output_b_task, output_e_task, alpha=self.opt['beta'])

        l_total = 0
        if self.cpn_loss_func is not None :
            loss, log = self.cpn_loss_func(self.former_proto_list.detach(),
                                           self.former_proto_label.detach(),
                                           self.output,
                                           self.labels)

            self.log_dict.update(log)
            self.log = log
            l_total += loss

        l_total.backward(retain_graph=False)
        self.optimizer_g.step()
        self.optimizer_e.step()

    def scale_output(self, output, output_e, alpha=0.5):
        self.output = output * alpha + output_e * (1-alpha)

    def incremental_fine_tune(self, train_dataset, train_loader, val_dataset, val_loader,
                              num_epoch, task_id=-1, test_id=-1, tb_logger=None, mask=None):
        """
        fine tune the models with the samples of incremental novel class

        Args:
            train_dataset (torch.utils.data.Dataset): the training dataset
            val_dataset (torch.utils.data.Dataset): the validation dataset
            num_epoch (int): the number of epoch to fine tune the models
            task_id (int): the id of sessions
            test_id (int): the id of few-shot test
        """

        current_iter = 0
        for epoch in range(num_epoch):
            self.epoch = epoch
            for idx, data in enumerate(train_loader):
                current_iter += 1
                self.update_learning_rate(
                    current_iter, warmup_iter=-1)

                with torch.no_grad():
                    self.feed_data(data)

                self.incremental_optimize_parameters(current_iter, mask=mask)

                if (idx % 10) == 0:
                    loss = self.log['PTFixPNLoss']
                    acc = self.log['Acc']
                    print(f'sess:{self.now_session_id}, epoch:{epoch}, loss:{loss}, acc:{acc}')

    def __NCM_incremental_test(self, test_dataset, task_id=-1, test_id=-1):
        prototypes = []
        pt_labels = []
        for key, value in self.prototypes_dict.items():
            prototypes.append(value)
            pt_labels.append(key)

        prototypes = torch.stack(prototypes).cuda()
        pt_labels = torch.tensor(pt_labels).cuda()

        if self.opt.get('details', False):
            data_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, drop_last=False)
        else:
            data_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=False,
                                                      drop_last=False)

        acc_ave = Averager()
        acc_former_ave = Averager()
        acc_former_all_ave = Averager()
        acc_novel_all_ave = Averager()

        for idx, data in enumerate(data_loader):
            self.feed_data(data)
            self.test()

            pairwise_distance = pair_euclidean_distances(self.output, prototypes)

            estimate = torch.argmin(pairwise_distance, dim=1)

            estimate_labels = pt_labels[estimate]

            acc = (estimate_labels ==
                   self.labels).sum() / float(estimate_labels.shape[0])

            acc_ave.add(acc.item(), int(estimate_labels.shape[0]))

            if self.opt.get('details', False):
                if self.labels.item() < self.opt['train']['bases']:
                    acc_former_all_ave.add(acc.item(), int(estimate_labels.shape[0]))
                else:
                    acc_novel_all_ave.add(acc.item(), int(estimate_labels.shape[0]))

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

    def get_prototypes(self, training_dataset):
        """
        calculated the prototypes for each class in training dataset

        Args:
            training_dataset (torch.utils.data.Dataset): the training dataset

        Returns:
            tuple: (prototypes_list, labels_list) where prototypes_list is the list of prototypes and
            labels_list is the list of class labels
        """
        training_dataset.set_aug(flag=False)
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
        selected_classes = get_session_classes(
            session=self.now_session_id,
            opt=self.opt)

        if self.now_session_id > 0:
            novel_classes = selected_classes[-self.opt['train']['way']:]
        else:
            novel_classes = selected_classes

        for cl in novel_classes:
            index_cl = torch.where(cl == labels)[0]
            class_features = features[index_cl]
            if self.use_cosine:
                class_features = F.normalize(class_features, dim=1)
            prototypes_list.append(class_features.mean(dim=0))

        prototypes_list = torch.stack(prototypes_list, dim=0)
        # reset augmentation
        training_dataset.set_aug(flag=True)

        return prototypes_list, novel_classes

    def feed_data(self, data):
        """
        The Data structure is (images, labels, labels_softmax)
        """

        self.images = data[0].cuda()
        self.labels = data[1].cuda()

        try:
            self.labels_softmax = data[2].cuda()
        except:
            self.labels_softmax = self.labels

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        original_output = self.net_g.forward(self.images)

        l_total = 0

        self.log_dict = AvgDict()

        loss = self.loss_func(original_output, self.labels_softmax)
        log_dict = {'CELoss': loss.item()}
        self.log_dict.add_dict(log_dict)
        l_total += loss

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.log_dict.get_ordinary_dict()

    def test(self, mask=None, mode='test'):
        self.net_g.eval()
        self.net_e.eval()
        with torch.no_grad():
            # ======================AANet===========================
            self.output, self.output_fc = self.net_g.forward_score(
                self.images)

            self.output_e, self.output_e_fc = self.net_e.forward_score(
                self.images)

            if self.now_session_id > 0:
                self.scale_output(self.output, self.output_e, alpha=self.opt['beta'])


    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, name=''):
        logger = get_root_logger()
        logger.info('Only support single GPU validation.')
        acc = self.nondist_validation(dataloader, current_iter, tb_logger, save_img, name)
        return acc

    def nondist_validation(self, training_dataset, dataloader, current_iter, tb_logger, name='', mask=None):
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

        return acc

    def __nondist_validation(self, training_dataset, dataloader):
        acc_ave = Averager()

        for idx, data in enumerate(dataloader):
            self.feed_data(data)
            self.test(output_embedding=False)

            #print("te_base_labels:{}".format(data[1].max()))
            estimate_labels = torch.argmax(self.output, dim=1)
            acc = (estimate_labels ==
                   self.labels_softmax).sum() / float(estimate_labels.shape[0])

            acc_ave.add(acc.item(), int(self.labels_softmax.shape[0]))

        # tentative for out of GPU memory
        del self.images
        del self.labels
        del self.labels_softmax
        del self.output
        del self.output_fc

        # set model to be trainable
        self.net_g.train()
        self.net_e.train()
        return acc_ave.item()

    def save(self, epoch, current_iter, name='net_g', dataset=None, mask=None,
             task_id=None):
        self.save_network(self.net_g, name, current_iter)
        self.save_network(self.net_e, name + '_extra', current_iter)

        self.save_training_state(epoch, current_iter)
        if self.is_incremental:
            self.save_prototypes(self.now_session_id, self.opt['test_id'])

    def load_prototypes(self, session_id, test_id):
        if session_id >= 0:
            if self.opt['train']['novel_exemplars'] == 0:
                load_filename = f'test{0}_session{session_id}.pt'
            else:
                load_filename = f'test{0}_session{0}.pt'
            load_path = osp.join(self.opt['path']['prototypes'], load_filename)
            print(f' --------------------------------------------------------')
            print(f' ----load_prototypes:{load_path}')
            print(f' --------------------------------------------------------')
            prototypes_dict = torch.load(load_path)
            self.prototypes_dict = prototypes_dict
            self.former_proto_list, self.former_proto_label = self._read_prototypes()
        else:
            if self.opt['path'].get('pretrain_prototypes', None) is not None:
                load_path = self.opt['path']['pretrain_prototypes']
                print(f' --------------------------------------------------------')
                print(f' ----load_pretrain_prototypes:{load_path}')
                print(f' --------------------------------------------------------')
                prototypes_dict = torch.load(self.opt['path']['pretrain_prototypes'])
                self.prototypes_dict = prototypes_dict
                self.former_proto_list, self.former_proto_label = self._read_prototypes()

    def save_prototypes(self, session_id, test_id):
        if session_id >= 0:
            save_path = osp.join(self.opt['path']['prototypes'], f'test{test_id}_session{session_id}.pt')
            torch.save(self.prototypes_dict, save_path)

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
