import importlib
import torch
import torch.nn as nn
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
import math
import os

from torch.nn import init as init
from torch.nn import functional as F

from methods.iCaRL_model import ICaRLModel
from methods.losses import InterClassSeparation, LessForget
from utils import get_root_logger, Averager, dir_size, one_hot
from data.normal_dataset import NormalDataset
from metrics import pair_euclidean_distances

loss_module = importlib.import_module('methods.losses')

class RebalanceModel(ICaRLModel):
    """Metric-based learning model"""
    def init_training_settings(self):
        train_opt = self.opt['train']

        if self.is_incremental:
            self.now_session_id = self.opt['task_id'] + 1
            self.num_novel_class = train_opt['num_class_per_task'] if self.now_session_id > 0 else 0
            self.total_class = train_opt['bases'] + self.num_novel_class * self.now_session_id
            self.num_old_class = self.total_class - self.num_novel_class if self.now_session_id > 0 else self.total_class

        # load base models for incremental learning
        load_base_model_path = self.opt['path'].get('base_model', None)
        if load_base_model_path is not None and self.is_incremental:
            self.load_network(self.net_g, load_base_model_path,
                              self.opt['path']['strict_load'])

            # record the former network
            self.net_g_former = deepcopy(self.net_g)
            self.net_g_former.eval()

        self.net_g.train()

        # define the weight matrix of classifier
        load_base_cf_path = self.opt['path'].get('base_model_cf', None)
        if load_base_cf_path is None and self.is_train:
            num_classes = self.opt['network_g']['num_classes']
            embed_dim = self.opt['train']['Embed_dim']
            self.cf_matrix = torch.empty(num_classes, embed_dim, requires_grad=True, device='cuda')
            self.sigma = torch.tensor(1.0, requires_grad=True, device='cuda')
            stdv = 1. / math.sqrt(self.cf_matrix.shape[1])
            init.uniform_(self.cf_matrix, -stdv, stdv)
            self.cf_matrix = OrderedDict([('fc', self.cf_matrix)])

        elif load_base_cf_path is not None and self.is_incremental:
            former_cf = torch.load(load_base_cf_path)
            self.former_cf_matrix = former_cf[0]
            self.former_cf_matrix.requires_grad = False

            self.sigma = former_cf[1]

            embed_dim = self.opt['train']['Embed_dim']

            if self.now_session_id > 0:
                self.cf_matrix_novel = torch.empty(self.num_novel_class, embed_dim, requires_grad=True, device='cuda')
                stdv = 1. / math.sqrt(self.cf_matrix_novel.shape[1])
                init.uniform_(self.cf_matrix_novel, -stdv, stdv)

                self.cf_matrix = OrderedDict([('fc.former', self.former_cf_matrix),('fc.novel', self.cf_matrix_novel)])
            else:
                self.cf_matrix = OrderedDict([('fc.former', self.former_cf_matrix)])

        else:
            raise ValueError('For incremental procedure, the classifier path `base_model_cf` has to be provided;'
                             'For training procedure, the classifier path `base_model_cf` should be None')

        # define losses
        self.loss_func = nn.CrossEntropyLoss()

        if train_opt.get('noise_loss'):
            noise_loss_type = train_opt['noise_loss'].pop('type')
            noise_loss_func = getattr(loss_module, noise_loss_type)
            self.noise_loss_func = noise_loss_func(**train_opt['noise_loss']).cuda()
        else:
            self.noise_loss_func = None

        if train_opt.get('proto_loss'):
            proto_loss_type = train_opt['proto_loss'].pop('type')
            proto_loss_func = getattr(loss_module, proto_loss_type)
            self.proto_loss_func = proto_loss_func(**train_opt['proto_loss']).cuda()
        else:
            self.proto_loss_func = None

        self.loss_LessForget = None
        self.loss_InterClassSeparation = None
        if self.is_incremental and self.now_session_id > 0:
            for loss_opt in train_opt['loss']:
                type = loss_opt.pop('type')
                if type == 'CosineCrossEntropy':
                    self.loss_func = nn.CrossEntropyLoss()
                if type == 'LessForget':
                    cur_lambda = loss_opt['lambda_base'] * math.sqrt(float(self.num_old_class)/self.num_novel_class)
                    self.loss_LessForget = LessForget(lmbda=cur_lambda)
                if type == 'InterClassSeparation':
                    self.loss_InterClassSeparation = InterClassSeparation(**loss_opt)

        # define the buffer
        if self.is_incremental and self.now_session_id > 0:
            self.img_buffer = self._load_img_buffer()
        else:
            self.img_buffer = None

        # setup the parameters of schedulers
        self.setup_schedulers_params()
        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def incremental_init(self, train_set, val_set):
        """ Initializing the incremental learning procedure

        Args:
            train_set (torch.utils.data.Dataset): the training dataset
            val_set (torch.utils.data.Dataset): the validation dataset
        """
        #prototypes, labels = self.get_prototypes(train_set)

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

        batch_size = self.opt['datasets']['train']['batch_size']
        self.img_buffer.combine_another_dataset(train_dataset)
        train_loader = torch.utils.data.DataLoader(dataset=self.img_buffer, num_workers=4, batch_size=batch_size, shuffle=True, drop_last=True)

        current_iter = 0
        for epoch in range(num_epoch):
            for idx1, data in enumerate(train_loader):
                current_iter += 1
                self.update_learning_rate(
                    current_iter, warmup_iter=-1)

                self.feed_buffer_data(data)

                self.feed_data(data)

                self.incremental_optimize_parameters(current_iter)

                if current_iter % self.opt['logger']['print_freq'] == 0:
                    logger = get_root_logger()
                    message = f'[epoch:{epoch:3d}, iter:{current_iter:4,d}, lr:({self.get_current_learning_rate()[0]:.3e})] [ '
                    for key, value in self.log_dict.items():
                        message += f'{key}: {value:.4f}, '
                    logger.info(message + ']')

                if self.opt['val']['val_freq'] is not None and current_iter % self.opt[
                    'val']['val_freq'] == 0:
                    logger = get_root_logger()
                    log_str = f'Epoch {epoch}, Iteration {current_iter}, Validation Step:\n'
                    logger.info(log_str)

                    if not self.opt['val']['test_type'] == 'NCM':
                        self.incremental_update(train_dataset)

                    acc = self.incremental_test(val_dataset, task_id=task_id, test_id=test_id)


    def incremental_test(self, test_dataset, task_id=-1, test_id=-1):
        if self.opt['val']['test_type'] == 'CNN':
            acc = self.CNN_test(test_dataset, task_id=task_id, test_id=test_id)
            return acc

        prototypes = []
        pt_labels = []
        for key, value in self.prototypes_dict.items():
            prototypes.append(value)
            pt_labels.append(key)

        prototypes = torch.stack(prototypes).cuda()
        pt_labels = torch.tensor(pt_labels).cuda()

        data_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=8, shuffle=False, drop_last=False)

        acc_ave = Averager()
        for idx, data in enumerate(data_loader):
            self.feed_data(data)
            self.test_without_cf()

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

        log_str = f'Test_acc of task {task_id} on test {test_id} is \t {acc_ave.item():.5f}\n'
        logger = get_root_logger()
        logger.info(log_str)


        return acc_ave.item()

    def CNN_test(self, test_dataset, task_id=-1, test_id=-1):
        data_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=False, drop_last=False)

        acc_ave = Averager()
        for idx, data in enumerate(data_loader):
            self.feed_data(data)
            self.test()

            estimate_labels = torch.argmax(self.output, dim=1)
            acc = (estimate_labels ==
                   self.labels_softmax).sum() / float(estimate_labels.shape[0])

            acc_ave.add(acc.item(), int(self.labels_softmax.shape[0]))

        # tentative for out of GPU memory
        del self.images
        del self.labels
        del self.labels_softmax
        del self.output
        torch.cuda.empty_cache()

        log_str = f'Test_acc of task {task_id} on test {test_id} is \t {acc_ave.item():.5f}\n'
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

        if isinstance(self.cf_matrix, dict):
            lr_cf = train_opt['optim_g'].get('lr_cf', None)
            if lr_cf is not None:
                train_opt['optim_g'].pop('lr_cf')
                cf_params = []
                for k, v in self.cf_matrix.items():
                    if v.requires_grad:
                        cf_params.append(v)
                    else:
                        logger = get_root_logger()
                        logger.warning(f'Params {k} will not be optimized.')

                if self.sigma.requires_grad:
                    optim_params.append(self.sigma)

                optim_params = [{'params': optim_params},
                                {'params': cf_params, 'lr': lr_cf}]
            else:
                for k, v in self.cf_matrix.items():
                    if v.requires_grad:
                        optim_params.append(v)
                    else:
                        logger = get_root_logger()
                        logger.warning(f'Params {k} will not be optimized.')
                if self.sigma.requires_grad:
                    optim_params.append(self.sigma)


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

    def incremental_optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.net_g.train()
        self.output = self.net_g(self.images)

        current_feat = self.output
        self.output = self.norm_cosine_similarity(self.output, weight_matrix=self.cf_matrix, sigma=self.sigma)

        loss = 0.0
        if self.loss_LessForget is not None:
            self.test_former_embedding(norm=False)
            former_feat = self.buffer_output
            loss2, log = self.loss_LessForget(former_feat=former_feat, current_feat=current_feat)
            self.log_dict.update(log)
            loss += loss2

        loss1 = self.loss_func(self.output, self.labels_softmax)
        loss += loss1
        self.log_dict['CELoss'] = loss1.item()

        if self.loss_InterClassSeparation is not None:
            loss3, log = self.loss_InterClassSeparation(scores=self.scores_before_scalar, labels=self.labels_softmax, num_old_classes=self.num_old_class)
            self.log_dict.update(log)
            loss += loss3

        self.log_dict['Loss'] = loss.item()

        loss.backward()
        del self.scores_before_scalar
        self.optimizer_g.step()


    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.net_g.train()

        original_embedding = self.net_g(self.images)
        output = self.norm_cosine_similarity(original_embedding, weight_matrix=self.cf_matrix, sigma=self.sigma)

        l_total = 0.0
        loss = self.loss_func(output, self.labels_softmax)
        l_total += loss
        self.log_dict['Loss'] = loss.item()

        loss.backward()
        del self.scores_before_scalar
        self.optimizer_g.step()

    def test(self, output_embedding=False):
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(self.images)
            if output_embedding:
                self.embeddings = self.output
            self.output = self.norm_cosine_similarity(self.output, weight_matrix=self.cf_matrix, sigma=self.sigma)
        self.net_g.train()

    def test_without_cf(self, norm=False):
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(self.images)
            if norm:
                self.output = F.normalize(self.output, dim=1)
        self.net_g.train()

    def test_former_embedding(self, norm=False):
        self.net_g_former.eval()
        with torch.no_grad():
            self.buffer_output = self.net_g_former(self.buffer_images)
            if norm:
                self.buffer_output = F.normalize(self.buffer_output, dim=1)

    def norm_cosine_similarity(self, output, weight_matrix, sigma=None):
        """
        calculated the cosine normalization

        Args:
            output (torch.Tensor): the size of output is (batch_size, feat_dim)
            weight_matrix (dict of torch.Tensor): the size of weight matrix is (num_classes, feat_dim)
            sigma (torch.Tensor): learnable scalar. The size is (1,)
        Returns:
            results (torch.Tensor): the size of results is (batch_size, num_classes)
        """

        norm_output = F.normalize(output, dim=1)
        results = []
        for name, w in weight_matrix.items():
            results.append(F.linear(norm_output, F.normalize(w, dim=1)))

        results = torch.cat(results, dim=1)

        self.scores_before_scalar = results

        if sigma is not None:
            results = sigma * results
        # weight_matrix = weight_matrix.T.unsqueeze(0).expand(batch_size, num_classes, -1)
        # output = output.unsqueeze(1).expand(batch_size, num_classes, -1)
        #
        # cosine_similarity = nn.CosineSimilarity(dim=2)
        # results = cosine_similarity(output, weight_matrix)

        return results

    def _save_classifier(self, task_id, name='net_g'):
        if task_id == -1:
            task_id = 'latest'
        save_filename_cf = f'{name}_classifier_{task_id}.pth'
        save_path_cf = osp.join(self.opt['path']['models'], save_filename_cf)
        cf_matrix = torch.cat([v for k, v in self.cf_matrix.items()], dim=0)
        torch.save([cf_matrix, self.sigma], save_path_cf)

