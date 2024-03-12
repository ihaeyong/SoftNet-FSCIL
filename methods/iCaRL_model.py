import importlib
import torch
import torch.nn as nn
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
import os

from torch.nn import init as init
from methods.base_model import BaseModel
from methods import networks as networks
from utils import get_root_logger, Averager, dir_size, one_hot, mkdir_or_exist
from data.normal_dataset import NormalDataset
from metrics import pair_euclidean_distances
import wandb

loss_module = importlib.import_module('methods.losses')

class ICaRLModel(BaseModel):

    def __init__(self, opt):
        super(ICaRLModel, self).__init__(opt)
        self.use_cosine = self.opt.get('use_cosine', False)

        # define network
        self.net_g = networks.define_net_g(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        if not self.is_incremental:
            self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_model_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path']['strict_load'])

        if self.is_train or self.is_incremental:
            self.init_training_settings()

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
            self.cf_matrix = torch.empty(embed_dim, num_classes, requires_grad=True, device='cuda')
            init.kaiming_normal_(self.cf_matrix, nonlinearity='relu', mode='fan_out')

        elif load_base_cf_path is not None and self.is_incremental:
            self.former_cf_matrix = torch.load(load_base_cf_path)
            num_novel_classes = self.opt['train']['num_class_per_task']
            embed_dim = self.opt['train']['Embed_dim']

            if self.now_session_id > 0:
                self.cf_matrix_novel = torch.empty(embed_dim, num_novel_classes, requires_grad=True, device='cuda')
                init.kaiming_normal_(self.cf_matrix_novel, nonlinearity='relu', mode='fan_out')
                self.cf_matrix = torch.cat((self.former_cf_matrix, self.cf_matrix_novel), dim=1)
                # change the matrix to leaf tensor
                self.cf_matrix = torch.tensor(self.cf_matrix.tolist(), requires_grad=True, device='cuda')
            else:
                self.cf_matrix = self.former_cf_matrix
            # self.former_cf_matrix = self.model_to_device(self.former_cf_matrix)
            # self.cf_matrix = self.model_to_device(self.cf_matrix)
        else:
            raise ValueError('For incremental procedure, the classifier path `base_model_cf` has to be provided;'
                             'For training procedure, the classifier path `base_model_cf` should be None')

        # define losses
        if self.is_incremental:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            # self.loss_func = nn.BCEWithLogitsLoss()
            self.loss_func = nn.CrossEntropyLoss()
        # define the buffer
        if self.is_incremental and self.now_session_id > 0:
            self.img_buffer = self._load_img_buffer()
        else:
            self.img_buffer = None

        # set up optimizers and schedulers
        self.setup_schedulers_params()
        self.setup_optimizers()
        self.setup_schedulers()

    def incremental_init(self, train_set, val_set):
        """ Initializing the incremental learning procedure
        Args:
            train_set (torch.utils.data.Dataset): the training dataset
            val_set (torch.utils.data.Dataset): the validation dataset
        """
        self.total_class = self.opt['train']['bases'] +self.opt['train']['num_class_per_task'] * self.now_session_id

    def setup_schedulers_params(self):
        train_opt = self.opt['train']
        if self.opt['train']['scheduler']['type'] is None:
            return
        if self.is_incremental:
            if self.now_session_id > 0:
                total_images = self.num_old_class * train_opt['buffer_size'] + train_opt['shots'] * self.num_novel_class
                batch_size = train_opt['fine_tune_batch']

                iteration_per_epoch = int(total_images / batch_size)

                for key, value in train_opt['scheduler'].items():
                    if isinstance(value, list):
                        train_opt['scheduler'][key] = [iteration_per_epoch * epoch for epoch in value]

    def incremental_update(self, novel_dataset, buffer_dataset=None):
        train_opt = self.opt['val']

        test_type = train_opt.get('test_type', 'NCM')

        if test_type == 'NCM' or self.now_session_id == 0:
            if self.opt['train'].get('calculate_pt_with_buffer', False):
                if self.now_session_id == 0:
                    self.obtain_buffer_index(novel_dataset)
                    novel_dataset.sample_the_buffer_data_with_index(self.sample_index)

            prototypes_list, labels_list = self.get_prototypes(novel_dataset)
            # update prototypes dict
            for i in range(prototypes_list.shape[0]):
                self.prototypes_dict.update({labels_list[i].item(): prototypes_list[i]})
            if buffer_dataset is not None:
                prototypes_list, labels_list = self.get_prototypes(buffer_dataset)
                # update prototypes dict
                for i in range(prototypes_list.shape[0]):
                    self.prototypes_dict.update({labels_list[i].item(): prototypes_list[i]})

    def incremental_test(self, test_dataset, task_id=-1, test_id=-1):
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
                self.test_former_classifier()

                self.feed_data(data)

                self.optimize_parameters(current_iter)

                # del self.images
                # del self.labels
                # del self.output
                # del self.labels_softmax
                # del self.buffer_images
                # del self.buffer_labels
                # del self.buffer_output
                # del self.buffer_labels_softmax
                # torch.cuda.empty_cache()

                logger = get_root_logger()
                if current_iter % self.opt['logger']['print_freq'] == 0:

                    message = f'[epoch:{epoch:3d}, iter:{current_iter:4,d}, lr:({self.get_current_learning_rate()[0]:.3e})]'
                    loss = self.log_dict['Loss']
                    message += f'[loss: {loss}]'

                    logger.info(message)


            if self.opt['val']['val_freq'] is not None and (epoch + 1)% self.opt[
                'val']['val_freq'] == 0:

                self.incremental_update(self.img_buffer)
                log_str = f'Epoch {epoch}, Iteration {current_iter}, Validation Step:\n'
                logger = get_root_logger()
                logger.info(log_str)
                acc = self.incremental_test(val_dataset, task_id=task_id, test_id=test_id)

            if (epoch + 1) == num_epoch:
                self.incremental_update(self.img_buffer)


    def nondist_validation(self, training_dataset, dataloader, current_iter, tb_logger, name=''):
        """
        Args:
            current_iter: the current iteration. -1 means the testing procedure
        """
        self.net_g.eval()
        acc = self.__nondist_validation(dataloader)
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

    def __nondist_validation(self, dataloader):
        acc_ave = Averager()

        for idx, data in enumerate(dataloader):
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

        if self.cf_matrix.requires_grad:
            lr_cf = train_opt['optim_g'].get('lr_cf', None)
            if lr_cf is not None:
                train_opt['optim_g'].pop('lr_cf')
                optim_params = [{'params': optim_params},
                                {'params': self.cf_matrix, 'lr': lr_cf}]
            else:
                optim_params.append(self.cf_matrix)

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

    def feed_buffer_data(self, data):
        """
        The Data structure is (images, labels, labels_softmax)
        """
        self.buffer_images = data[0].cuda()
        self.buffer_labels = data[1].cuda()
        self.buffer_labels_softmax = data[2].cuda()

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.net_g.train()
        self.output = self.net_g(self.images)
        self.output = self.output.matmul(self.cf_matrix)

        if self.is_train:
            # labels_one_hot = one_hot(self.labels_softmax, num_class=self.opt['network_g']['num_classes'])
            # loss = self.loss_func(self.output, labels_one_hot)
            loss = self.loss_func(self.output, self.labels_softmax)
        elif self.is_incremental:
            labels_one_hot = one_hot(self.labels_softmax, num_class=self.total_class)
            self.former_logits = torch.sigmoid(self.buffer_output)
            index = torch.tensor(range(self.former_logits.shape[1])).unsqueeze(0).expand(self.former_logits.shape[0], -1).cuda()
            labels_one_hot = labels_one_hot.scatter_(dim=1, index=index, src=self.former_logits)
            loss = self.loss_func(self.output, labels_one_hot)

        self.log_dict['Loss'] = loss.item()

        loss.backward()
        self.optimizer_g.step()

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(self.images)
            self.output = self.output.matmul(self.cf_matrix)
        self.net_g.train()

    def test_without_cf(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(self.images)
        self.net_g.train()

    def test_former_classifier(self):
        self.net_g_former.eval()
        with torch.no_grad():
            self.buffer_output = self.net_g_former(self.buffer_images)
            self.buffer_output = self.buffer_output.matmul(self.former_cf_matrix)

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
            self.test_without_cf()
            features_list.append(self.output)
            labels_list.append(self.labels)

            # tentative for out of GPU memory
            del self.images
            del self.labels
            del self.labels_softmax
            del self.output
            torch.cuda.empty_cache()

        features = torch.cat(features_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        selected_classes = training_dataset.selected_classes
        for cl in selected_classes:
            index_cl = torch.where(cl == labels)[0]
            class_features = features[index_cl]
            prototypes_list.append(class_features.mean(dim=0))

        prototypes_list = torch.stack(prototypes_list, dim=0)
        # reset augmentation
        training_dataset.set_aug(aug)
        return prototypes_list, torch.from_numpy(training_dataset.selected_classes)

    def save(self, epoch, current_iter, name='net_g', dataset=None):
        self.save_network(self.net_g, name, current_iter)
        self.save_training_state(epoch, current_iter)
        self._save_classifier(task_id=current_iter, name=name)
        if self.is_incremental:
            self._save_img_buffer(dataset, self.opt['test_id'])

    def set_the_saving_files_path(self, opt, task_id):
        test_id = self.opt['test_id']
        save_filename_g = f'test{test_id}_session_{task_id}.pth'
        save_filename_cf = f'test{test_id}_session_classifier_{task_id}.pth'
        save_path_g = osp.join(opt['path']['models'], save_filename_g)
        save_path_cf = osp.join(opt['path']['models'], save_filename_cf)
        opt['path']['base_model'] = save_path_g
        opt['path']['base_model_cf'] = save_path_cf

    def _save_img_buffer(self, dataset, test_id):
        if self.opt['train']['random']:
            dataset.sample_the_buffer_data(self.opt['train']['buffer_size'])
        else:
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

            dataset.sample_the_buffer_data_with_index(sample_index)

        classes, class_to_idx = dataset._find_classes(dataset.dataroot)
        # make dir for buffer
        for class_name in classes:
            target_path = osp.join(self.opt['path']['feat_buffer'], f'test{test_id}')
            mkdir_or_exist(target_path)
            target_path = osp.join(target_path, class_name)
            mkdir_or_exist(target_path)

        img_path_list = dataset.image_path_list
        for img_path in img_path_list:
            img_name = osp.basename(img_path)
            img_dir = osp.dirname(img_path)
            class_name = osp.split(img_dir)[1]
            target_path = osp.join(self.opt['path']['feat_buffer'], f'test{test_id}', class_name, img_name)
            os.system(f'cp {img_path} {target_path}')

    def _save_classifier(self, task_id, name='net_g'):
        if task_id == -1:
            task_id = 'latest'
        save_filename_cf = f'{name}_classifier_{task_id}.pth'
        save_path_cf = osp.join(self.opt['path']['models'], save_filename_cf)
        torch.save(self.cf_matrix, save_path_cf)

    def _load_img_buffer(self):
        now_buffer_size, num_imgs = dir_size(self.opt['path']['feat_buffer'])
        logger = get_root_logger()
        logger.info(f'The size of buffer is {float(now_buffer_size)/1024.0/1024.0} Mb, the number of images is {num_imgs}')

        bases = self.opt['train']['bases']
        num_class_per_task = self.opt['train']['num_class_per_task']
        selected_class = (self.opt['class_permutation'])[:bases + (self.now_session_id - 1) * num_class_per_task]

        test_id = self.opt['test_id']
        root = osp.join(self.opt['path']['feat_buffer'], f'test{test_id}')

        dataset_opt = {'dataroot': root, 'selected_classes': selected_class, 'aug': True}
        dataset_opt['transformer_agu'] = self.opt['datasets']['train']['transformer_agu']
        dataset_opt['transformer'] = self.opt['datasets']['train']['transformer']
        dataset_opt['all_classes'] = self.opt['class_permutation']
        dataset_opt['pre_load'] = False
        if self.opt['datasets']['train'].get('user_defined', False):
            dataset_opt['user_defined'] = True


        buffer = NormalDataset(dataset_opt)

        return buffer

    def _get_features(self, dataset):
        aug = dataset.get_aug()
        dataset.set_aug(False)

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=8, shuffle=False, drop_last=False)

        features = []
        labels = []
        for i, data in enumerate(data_loader, 0):
            self.feed_data(data)
            self.test_without_cf()
            features.append(self.output.cpu())
            labels.append(self.labels.cpu())

            del self.images
            del self.labels
            del self.labels_softmax
            del self.output
            torch.cuda.empty_cache()

        dataset.set_aug(aug)

        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)
        return features, labels