import logging
import os
import torch
from torch.distributions.uniform import Uniform
import torchvision
import numpy as np
from utils import BetaDistribution, BoundUniform, BoundNormal
from utils import BetaDistribution, BoundUniform, BoundNormal, DiscreteUniform, \
    DiscreteUniform2, DiscreteBetaDistribution
from collections import OrderedDict
from copy import deepcopy
from torch.nn.parallel import DataParallel, DistributedDataParallel

from methods import lr_scheduler as lr_scheduler

logger = logging.getLogger('FS-IL')


class BaseModel():
    """Base model."""

    def __init__(self, opt):
        self.opt = opt
        self.is_train = opt['is_train']
        self.schedulers = []
        self.optimizers = []
        self.log_dict = {}
        self.wandb_logger = opt['wandb_logger']

        #incremental setting
        self.is_incremental = opt['is_incremental']
        self.prototypes_dict = {}

        #feat buffer
        self.feat_buffer = []
        self.labels_buffer = []

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def init_training(self, dataset):
        pass

    def save(self, epoch, current_iter, name='net_g', dataset=None):
        """Save networks and training state."""
        pass

    def setup_schedulers_params(self):
        """Setting the parameters of schedulers."""
        pass

    def extend_feat_buffer(self, dataset):
        """add features of samples in dataset to feature buffer

        Arg:
            dataset (torch.utils.data.Dataset): Training Set
        """
        pass

    def add_sample_to_buffer(self, feat, label):
        self.feat_buffer.append(feat)
        self.labels_buffer.append(label)

    def validation(self, train_set, dataloader, current_iter, tb_logger, name='', mask=None):
        """Validation function.

        Args:
            train_set (torch.utils.data.DataSet): Dataset for training, is used to obtain the prototypes
            dataloader (torch.utils.data.DataLoader): Validation dataloader.
            current_iter (int): Current iteration.
            tb_logger (tensorboard logger): Tensorboard logger.
        """
        # TODO distributed learning
        #if self.opt['dist']:

        #    self.dist_validation(dataloader, current_iter, tb_logger)
        #else:
        acc = self.nondist_validation(train_set, dataloader, current_iter, tb_logger, mask=mask)
        return acc

    def get_current_log(self):
        return self.log_dict

    def model_to_device(self, net, device):
        """Model to device. It also warps models with DistributedDataParallel
        or DataParallel.

        Args:
            net (nn.Module)
        """
        net = net.to(device)

        # TODO (SGY) support distributed learning
        # if self.opt['dist']:
        #     find_unused_parameters = self.opt.get('find_unused_parameters',
        #                                           False)
        #     net = DistributedDataParallel(
        #         net,
        #         device_ids=[torch.cuda.current_device()],
        #         find_unused_parameters=find_unused_parameters)
        # elif self.opt['num_gpu'] > 1:
        #     net = DataParallel(net)
        return net

    def setup_schedulers(self):
        """Set up schedulers."""
        #self.setup_schedulers_params()

        train_opt = self.opt['train']
        scheduler_type = train_opt['scheduler'].pop('type')
        if scheduler_type is None:
            return
        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.MultiStepRestartLR(optimizer,
                                                    **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingRestartLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingRestartLR(
                        optimizer, **train_opt['scheduler']))
        else:
            raise NotImplementedError(
                f'Scheduler {scheduler_type} is not implemented yet.')
        train_opt['scheduler']['type'] = scheduler_type

    def get_bare_model(self, net):
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module
        return net

    def print_network(self, net):
        """Print the str and parameter number of a network.

        Args:
            net (nn.Module)
        """
        #if isinstance(net, (DataParallel, DistributedDataParallel)):
        #    net_cls_str = (f'{net.__class__.__name__} - '
        #                   f'{net.module.__class__.__name__}')
        #else:

        net_cls_str = f'{net.__class__.__name__}'

        net = self.get_bare_model(net)
        net_str = str(net)
        net_params = sum(map(lambda x: x.numel(), net.parameters()))

        logger.info(
            f'Network: {net_cls_str}, with parameters: {net_params:,d}')
        logger.info(net_str)

    def _set_lr(self, lr_groups_l):
        """Set learning rate for warmup.
        Args:
            lr_groups_l (list): List for lr_groups, each for an optimizer.
        """
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group['lr'] = lr

    def _get_init_lr(self):
        """Get the initial lr, which is set by the scheduler.
        """
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append(
                [v['initial_lr'] for v in optimizer.param_groups])
        return init_lr_groups_l

    def update_learning_rate(self, current_iter, warmup_iter=-1):
        """Update learning rate.
        Args:
            current_iter (int): Current iteration.
            warmup_iter (int)： Warmup iter numbers. -1 for no warmup.
                Default： -1.
        """
        if current_iter > 1:
            for scheduler in self.schedulers:
                scheduler.step()
        # set up warm-up learning rate
        if current_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            # currently only support linearly warm up
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append(
                    [v / warmup_iter * current_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)

    def get_current_learning_rate(self):
        return [
            param_group['lr']
            for param_group in self.optimizers[0].param_groups
        ]

    def save_mask(self, data, task_id, net_label, current_iter, debug=False):
        if current_iter == -1:
            current_iter = 'latest'
        save_filename = f'{net_label}_subnet{task_id}_{current_iter}.pth'
        save_path = os.path.join(self.opt['path']['models'], save_filename)
        torch.save(data, save_path)

        if debug:
            print("Saved {}".format(save_path))

    def save_network(self, net, net_label, current_iter, param_key='params'):
        """Save networks.
        Args:
            net (nn.Module | list[nn.Module]): Network(s) to be saved.
            net_label (str): Network label.
            current_iter (int): Current iter number.
            param_key (str | list[str]): The parameter key(s) to save network.
                Default: 'params'.
        """
        if current_iter == -1:
            current_iter = 'latest'
        save_filename = f'{net_label}_{current_iter}.pth'
        save_path = os.path.join(self.opt['path']['models'], save_filename)

        net = net if isinstance(net, list) else [net]
        param_key = param_key if isinstance(param_key, list) else [param_key]
        assert len(net) == len(
            param_key), 'The lengths of net and param_key should be the same.'

        save_dict = {}
        for net_, param_key_ in zip(net, param_key):
            net_ = self.get_bare_model(net_)
            state_dict = net_.state_dict()
            for key, param in state_dict.items():
                if key.startswith('module.'):  # remove unnecessary 'module.'
                    key = key[7:]
                state_dict[key] = param.cpu()
            save_dict[param_key_] = state_dict

        torch.save(save_dict, save_path)

    def _print_different_keys_loading(self, crt_net, load_net, strict=True):
        """Print keys with differnet name or different size when loading models.
        1. Print keys with differnet names.
        2. If strict=False, print the same key but with different tensor size.
            It also ignore these keys with different sizes (not load).
        Args:
            crt_net (torch model): Current network.
            load_net (dict): Loaded network.
            strict (bool): Whether strictly loaded. Default: True.
        """
        crt_net = self.get_bare_model(crt_net)
        crt_net = crt_net.state_dict()
        crt_net_keys = set(crt_net.keys())
        load_net_keys = set(load_net.keys())

        if crt_net_keys != load_net_keys:
            logger.warning('Current net - loaded net:')
            for v in sorted(list(crt_net_keys - load_net_keys)):
                logger.warning(f'  {v}')
            logger.warning('Loaded net - current net:')
            for v in sorted(list(load_net_keys - crt_net_keys)):
                logger.warning(f'  {v}')

        # check the size for the same keys
        if not strict:
            common_keys = crt_net_keys & load_net_keys
            for k in common_keys:
                if crt_net[k].size() != load_net[k].size():
                    logger.warning(
                        f'Size different, ignore [{k}]: crt_net: '
                        f'{crt_net[k].shape}; load_net: {load_net[k].shape}')
                    load_net[k + '.ignore'] = load_net.pop(k)

    def percentile(self, scores, sparsity):
        k = 1 + round(.01 * float(sparsity) * (scores.numel() - 1))
        return scores.view(-1).kthvalue(k).values.item()

    def get_subnet_mask(self, weight, sparsity=1.0):

        zeros = torch.zeros(weight.shape)
        ones = torch.ones(weight.shape)

        k_val = self.percentile(weight, sparsity*100)
        return torch.where(weight < k_val, zeros, ones)


    def load_base_network(self, net, load_path, strict=True, param_key='params', is_train=True):
        """Load network pretrained Imagenet

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network.
                Default: 'params'.
        """
        net = self.get_bare_model(net)
        logger.info(
            f'Loading {net.__class__.__name__} model from {load_path}.')

        if False:
            load_net = torch.load(
                load_path, map_location=lambda storage, loc: storage)[param_key]
        else:
            # multi-gpu learning
            load_net = torch.load(load_path, map_location='cuda')['state_dict']

        adopt_cls = self.opt['network_g'].get('adopt_classifier', None)
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            new_k = k.replace('module.', '')
            load_net[new_k] = load_net.pop(k)
            if k.startswith('func.fc.') :
                load_net.pop(k)
        self._print_different_keys_loading(net, load_net, strict)

        net.load_state_dict(load_net, strict=strict)

        if is_train:
            for name, param in net.named_parameters():
                if 'bn' in name and param is not None:
                    param.requires_grad=True
                elif 'fc' in name :
                    param.requires_grad=True
                elif 'w_m' in name :
                    param.requires_grad=True
                else:
                    param.requires_grad=True

                print(f'load pre-trained param to be trainable: {name}')


    def load_base_v1_network(self, net, load_path, strict=True, param_key='params', is_train=True):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network.
                Default: 'params'.
        """

        if True:
            net = self.get_bare_model(net)
            logger.info(
                f'Loading {net.__class__.__name__} model from {load_path}.')

            pre_net = torchvision.models.resnet18
            pre_net = pre_net(pretrained=True)
            pre_net = pre_net if isinstance(pre_net, list) else [pre_net]
            param_key = param_key if isinstance(param_key, list) else [param_key]

            save_dict = {}
            for net_, param_key_ in zip(pre_net, param_key):
                net_ = self.get_bare_model(net_)
                state_dict = net_.state_dict()
                for key, param in state_dict.items():
                    state_dict[key] = param.cpu()
                save_dict[param_key_] = state_dict

        if is_train:
            for name, param in net.named_parameters():
                name=name.replace('func.', '')
                if name in save_dict['params'].keys():
                    #if 'fc' in name:
                    #    continue

                    param = save_dict['params'][name].cuda()
                    if 'bn' in name:
                        key_split = name.replace('func.', '').split('.')
                        if len(key_split) == 2:
                            module_attr, module_name = key_split[1], key_split[0]
                            module = getattr(net.func, module_name)
                            module.running_mean = save_dict['params'][name.replace('weight', 'running_mean')].cuda()
                            module.running_var = save_dict['params'][name.replace('weight', 'running_var')].cuda()
                            module.num_batches_tracked = save_dict['params'][name.replace('weight', 'num_batches_tracked')].cuda()
                        else:
                            module = getattr(getattr(net.func, key_split[0])[int(key_split[1])], key_split[2])
                            module.running_mean = save_dict['params'][name.replace('weight', 'running_mean')].cuda()
                            module.running_var = save_dict['params'][name.replace('weight', 'running_var')].cuda()
                            module.num_batches_tracked = save_dict['params'][name.replace('weight', 'num_batches_tracked')].cuda()

                    if 'downsample.1' in name:
                        key_split = name.replace('func.', '').split('.')
                        module = getattr(getattr(net.func, key_split[0])[int(key_split[1])], key_split[2])
                        module[1].running_mean = save_dict['params'][name.replace('weight', 'running_mean')].cuda()
                        module[1].running_var = save_dict['params'][name.replace('weight', 'running_var')].cuda()
                        module[1].num_batches_tracked = save_dict['params'][name.replace('weight', 'num_batches_tracked')].cuda()

                    print('updated:{}'.format(name))
                else:
                    if 'w_m' in name:
                        param = save_dict['params'][name.replace('w_m','weight')].cuda()
                        print('updated:{}'.format(name))
                    else:
                        print('non-updated:{}'.format(name))

    def load_full_network(self, net, load_path, strict=True, param_key='params', is_train=False):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network.
                Default: 'params'.
        """
        net = self.get_bare_model(net)
        logger.info(
            f'Loading {net.__class__.__name__} model from {load_path}.')
        load_net = torch.load(
            load_path, map_location=lambda storage, loc: storage)[param_key]

        adopt_cls = self.opt['network_g'].get('adopt_classifier', None)
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('func.fc.') and adopt_cls is False:
                load_net.pop(k)
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        self._print_different_keys_loading(net, load_net, strict)
        net.load_state_dict(load_net, strict=strict)


    def load_network(self, net, load_path, strict=True, param_key='params', is_train=False):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network.
                Default: 'params'.
        """
        net = self.get_bare_model(net)
        logger.info(
            f'Loading {net.__class__.__name__} model from {load_path}.')
        load_net = torch.load(
            load_path, map_location=lambda storage, loc: storage)[param_key]

        adopt_cls = self.opt['network_g'].get('adopt_classifier', None)
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('func.fc.') and adopt_cls is False:
                load_net.pop(k)
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        self._print_different_keys_loading(net, load_net, strict)
        net.load_state_dict(load_net, strict=strict)

        if is_train:
            for name, param in net.named_parameters():
                # 'bn' of downsampling
                if 'downsample.1.weight' in name:
                    name = name.replace('func.','')
                    name = name.replace('.weight', '')
                    key_split = name.split('.')
                    param.requires_grad=False
                    module = getattr(getattr(net.func, key_split[0])[int(key_split[1])], key_split[2])[int(key_split[3])]
                    module.affine = False
                    module.track_running_stats = False

                if 'bn' in name and param is not None:
                    param.requires_grad=False
                    key_split = name.replace('func.', '').split('.')
                    if len(key_split) == 2:
                        module_attr, module_name = key_split[1], key_split[0]
                        getattr(net.func, module_name).affine = False
                        getattr(net.func, module_name).track_running_stats = False
                    else:
                        module = getattr(getattr(net.func, key_split[0])[int(key_split[1])], key_split[2])
                        module.affine = False
                        module.track_running_stats = False

                elif 'fc' in name and adopt_cls is False:
                    param.requires_grad=True
                elif 'w_m' in name and param is not None:
                    param.requires_grad=True
                else:
                    param.requires_grad=True

    def load_cub_network(self, net, load_path, strict=True,
                         param_key='params', is_train=False):
        """Load network.

            Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network.
                Default: 'params'.
        """
        net = self.get_bare_model(net)
        logger.info(
            f'Loading {net.__class__.__name__} model from {load_path}.')
        load_net = torch.load(
            load_path, map_location=lambda storage, loc: storage)[param_key]

        adopt_cls = self.opt['network_g'].get('adopt_classifier', None)
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('func.fc.') and adopt_cls is False:
                load_net.pop(k)
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        self._print_different_keys_loading(net, load_net, strict)
        net.load_state_dict(load_net, strict=strict)

        if is_train:
            for name, param in net.named_parameters():

                if 'bn' in name:
                    key_split = name.replace('func.', '').split('.')
                    if len(key_split) == 2:
                        module_attr, module_name = key_split[1], key_split[0]
                        module = getattr(net.func, module_name)
                    else:
                        module = getattr(getattr(net.func, key_split[0])[int(key_split[1])], key_split[2])

                    module.affine = False
                    module.track_running_stats = False
                    print('fixed:{}'.format(name))

                if 'downsample.1' in name:
                    key_split = name.replace('func.', '').split('.')
                    module = getattr(getattr(net.func, key_split[0])[int(key_split[1])], key_split[2])
                    module[1].affine = False
                    module[1].track_running_stats = False

                    print('fixed:{}'.format(name))

    def save_training_state(self, epoch, current_iter):
        """Save training states during training, which will be used for
        resuming.

        Args:
            epoch (int): Current epoch.
            current_iter (int): Current iteration.
        """
        if current_iter != -1:
            state = {
                'epoch': epoch,
                'iter': current_iter,
                'optimizers': [],
                'schedulers': []
            }
            for o in self.optimizers:
                state['optimizers'].append(o.state_dict())
            for s in self.schedulers:
                state['schedulers'].append(s.state_dict())
                save_filename = f'{current_iter}.state'
                save_path = os.path.join(self.opt['path']['training_states'],
                                         save_filename)
                torch.save(state, save_path)

    def resume_training(self, resume_state):
        """Reload the optimizers and schedulers for resumed training.

        Args:
            resume_state (dict): Resume state.
        """
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(
            self.optimizers), 'Wrong lengths of optimizers'
        assert len(resume_schedulers) == len(
            self.schedulers), 'Wrong lengths of schedulers'
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)

    def reduce_loss_dict(self, loss_dict):
        # TODO
        """reduce loss dict.
        In distributed training, it averages the losses among different GPUs .
        Args:
            loss_dict (OrderedDict): Loss dict.
        """
        with torch.no_grad():
            if self.opt['dist']:
                keys = []
                losses = []
                for name, value in loss_dict.items():
                    keys.append(name)
                    losses.append(value)
                losses = torch.stack(losses, 0)
                torch.distributed.reduce(losses, dst=0)
                if self.opt['rank'] == 0:
                    losses /= self.opt['world_size']
                loss_dict = {key: loss for key, loss in zip(keys, losses)}

            log_dict = OrderedDict()
            for name, value in loss_dict.items():
                log_dict[name] = value.mean().item()

            return log_dict

    def save_original_range_params(self):
        self.origin_params = [param.data for param in self.net_g_first_params]

    def clamp_range_params(self):
        with torch.no_grad():
            for i in range(len(self.origin_params)):
                p_origin = self.origin_params[i]
                p = self.range_params[i]
                p_upper_bound = p_origin + self.bound_value / 2
                p_lower_bound = p_origin - self.bound_value / 2

                p_upper_mask = p.data > p_upper_bound
                p_lower_mask = p.data < p_lower_bound

                n_upper_pos = p_upper_mask.sum()
                n_lower_pos = p_lower_mask.sum()
                if n_upper_pos > 0 or n_lower_pos > 0:
                    logger.info(f'clamp range params {i}: {self.range_params_name[i]}, n_upper: {n_upper_pos}, n_lower: {n_lower_pos}')

                p.data[p_upper_mask] = p_upper_bound[p_upper_mask]
                p.data[p_lower_mask] = p_lower_bound[p_lower_mask]

    def generate_random_samplers(self, net):
        self.obtain_range_params(net)
        if self.random_noise is not None:
            self.samplers, self.bound_value = self._generate_random_samplers\
                (self.random_noise, self.range_params)

        if self.random_noise_val is not None:
            self.samplers_val, self.bound_value_val = self._generate_random_samplers\
                (self.random_noise_val, self.range_params_val)

    def obtain_range_params(self, net):
        train_opt = self.opt.get('train', None)
        val_opt = self.opt.get('val', None)
        self.random_noise = None
        self.random_noise_val = None

        if train_opt is not None:
            self.random_noise = train_opt.get('random_noise', None)
        if val_opt is not None:
            self.random_noise_val = val_opt.get('random_noise', None)

        if self.random_noise_val is not None:
            logger.info('Obtain parameters for val')

            self.random_times_val = self.random_noise_val['random_times']
            self.range_params_val, self.range_params_name_val = self.__obtain_range_params(self.random_noise_val, net)

        if self.random_noise is not None:
            logger.info('Obtain parameters for train')
            self.random_times = self.random_noise['random_times']
            self.range_params, self.range_params_name = self.__obtain_range_params(self.random_noise, net)

    def __obtain_range_params(self, random_noise, net):
        range_param = []
        range_param_name = []
        if random_noise['type'] == 'all_bias':
            for k, v in net.named_parameters():
                #logger.info(k)
                if k.endswith('bias') and k.find('fc') == -1:
                    range_param.append(v)
                    range_param_name.append(k)
        elif random_noise['type'] == 'all_weight':
            for k, v in net.named_parameters():
                #logger.info(k)
                if k.endswith('weight') and k.find('fc') == -1 and k.find('classifier') == -1:
                    range_param.append(v)
                    range_param_name.append(k)
        elif random_noise['type'] == 'all_conv_weight':
            for k, v in net.named_parameters():
                if k.endswith('weight') and k.find('conv') != -1:
                    range_param.append(v)
                    range_param_name.append(k)
        elif random_noise['type'] == 'suffix_conv_weight':
            num_layers = random_noise['num_layers']
            for k, v in net.named_parameters():
                if k.endswith('weight') and k.find('conv') != -1 and k.find('shortcut') == -1:
                    range_param.append(v)
                    range_param_name.append(k)

            range_param = range_param[-num_layers:]
            range_param_name = range_param_name[-num_layers:]

        elif random_noise['type'] == 'suffix_only_conv_weight':
            num_layers = random_noise['num_layers']
            for k, v in net.named_parameters():
                if k.endswith('weight') and k.find('conv') != -1 and k.find('shortcut') == -1 and k.find('w_m') == -1:
                    logger.info(k)
                    range_param.append(v)
                    range_param_name.append(k)

            range_param = range_param[-num_layers:]
            range_param_name = range_param_name[-num_layers:]

        elif random_noise['type'] == 'suffix_bn':
            num_layers = random_noise['num_layers']
            for k, v in net.named_parameters():
                if k.find('bn') != -1:
                    range_param.append(v)
                    range_param_name.append(k)
            range_param = range_param[-num_layers:]
            range_param_name = range_param_name[-num_layers:]
        elif random_noise['type'] == 'pre_bn':
            num_layers = random_noise['num_layers']
            for k, v in net.named_parameters():
                if k.find('bn') != -1:
                    range_param.append(v)
                    range_param_name.append(k)
            range_param = range_param[:num_layers]
            range_param_name = range_param_name[:num_layers]
        elif random_noise['type'] == 'all_bn':
            for k, v in net.named_parameters():
                if k.find('bn') != -1:
                    range_param.append(v)
                    range_param_name.append(k)
        elif random_noise['type'] == 'suffix_bias':
            num_layers = random_noise['num_layers']
            for k, v in net.named_parameters():
                #logger.info(k)
                if k.endswith('bias') and k.find('fc') == -1 and k.find('classifier') == -1:
                    range_param.append(v)
                    range_param_name.append(k)
            range_param = range_param[-num_layers:]
            range_param_name = range_param_name[-num_layers:]
        elif random_noise['type'] == 'prefix_bias':
            num_layers = random_noise['num_layers']
            for k, v in net.named_parameters():
                #logger.info(k)
                if k.endswith('bias') and k.find('fc') == -1 and k.find('classifier') == -1:
                    range_param.append(v)
                    range_param_name.append(k)
            range_param = range_param[:num_layers]
            range_param_name = range_param_name[:num_layers]

        elif random_noise['type'] == 'suffix_weight':
            num_layers = random_noise['num_layers']
            for k, v in net.named_parameters():
                #logger.info(k)
                if k.endswith('weight') and k.find('fc') == -1 and k.find('classifier') == -1:
                    range_param.append(v)
                    range_param_name.append(k)
            range_param = range_param[-num_layers:]
            range_param_name = range_param_name[-num_layers:]

        elif random_noise['type'] == 'prefix_weight':
            num_layers = random_noise['num_layers']
            for k, v in net.named_parameters():
                #logger.info(k)
                if k.endswith('weight') and k.find('fc') == -1 and k.find('classifier') == -1:
                    range_param.append(v)
                    range_param_name.append(k)
            range_param = range_param[:num_layers]
            range_param_name = range_param_name[:num_layers]
        elif random_noise['type'] == 'suffix_all':
            num_layers = random_noise['num_layers']
            for k, v in net.named_parameters():
                #logger.info(k)
                if k.find('fc') == -1 and k.find('classifier') == -1:
                    range_param.append(v)
                    range_param_name.append(k)
            range_param = range_param[-num_layers:]
            range_param_name = range_param_name[-num_layers:]

        return range_param, range_param_name

    def _generate_random_samplers(self, random_noise_opt, range_params, upper_bound=None, lower_bound=None):
        samplers = []
        distribution = random_noise_opt.get('distribution', None)
        assert distribution is not None

        type = distribution['type']
        bound_value = torch.tensor(random_noise_opt['bound_value'])
        for i, params in enumerate(range_params):
            if upper_bound is None:
                ub = torch.full(params.shape, bound_value / 2.0, device=torch.device('cuda'))
            else:
                ub = upper_bound[i]

            if lower_bound is None:
                lb = torch.full(params.shape, -bound_value / 2.0, device=torch.device('cuda'))
            else:
                lb = lower_bound[i]

            random_noise_opt['upper_bound'] = ub
            random_noise_opt['lower_bound'] = lb

            if type == 'Uniform':
                low = lb
                high = ub
                m = Uniform(low, high)
                samplers.append(m)
            elif type == 'DiscreteUniform':
                m = DiscreteUniform(bound_value, params.shape, random_noise_opt['reduction_factor'])
                samplers.append(m)
            elif type == 'DiscreteUniform2':
                m = DiscreteUniform2(bound_value, params.shape, random_noise_opt['reduction_factor'])
                samplers.append(m)
            # -----------------------------------------------------------------------------
            elif type == 'DiscreteBeta':
                low = random_noise_opt['low']
                high = random_noise_opt['high']
                r = random_noise_opt['reduction_factor']
                m = DiscreteBetaDistribution(low, high, params.shape, bound_value, r)
                samplers.append(m)
            elif type == 'Beta':
                alpha = distribution['alpha']
                beta = distribution['beta']
                alpha = torch.full(params.shape, alpha, device=torch.device('cuda'))
                beta = torch.full(params.shape, beta, device=torch.device('cuda'))
                bound = torch.full(params.shape, bound_value, device=torch.device('cuda'))
                m = BetaDistribution(alpha, beta, upper_bound=ub, lower_bound=lb)
                samplers.append(m)
            elif type == 'Random':
                alpha = [0.3, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
                beta = [0.3, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
                a_p = np.random.randint(7)
                b_p = np.random.randint(7)
                alpha = alpha[a_p]
                beta = beta[b_p]
                alpha = torch.full(params.shape, alpha, device=torch.device('cuda'))
                beta = torch.full(params.shape, beta, device=torch.device('cuda'))
                m = BetaDistribution(alpha, beta, upper_bound=ub, lower_bound=lb)
                samplers.append(m)
            elif type == 'RandomValueBeta':
                self.__Random_Value_Beta(params=params, samplers=samplers, random_noise_opt=random_noise_opt)
            elif type == 'RandomValueBeta2':
                self.__Random_Value_Beta2(params=params, samplers=samplers, random_noise_opt=random_noise_opt)
            elif type == 'RandomBeta':
                self.__Random_Beta(params=params, samplers=samplers, random_noise_opt=random_noise_opt)
            elif type == 'Random3':
                pair = [(0.8, 0.8), (0.8, 0.6), (0.6, 0.8), (0.7, 1.0), (1.0, 0.7), (1.0, 1.0), (1.0, 2.0), (2.0, 1.0), (2.0, 2.0), (1.5, 2.0), (2.0, 1.5), (1.0, 1.0)]
                pos = np.random.randint(len(pair))
                alpha = pair[pos][0]
                beta = pair[pos][1]
                alpha = torch.full(params.shape, alpha).cuda()
                beta = torch.full(params.shape, beta).cuda()
                bound = torch.full(params.shape, bound, device=torch.device('cuda'))
                m = BetaDistribution(alpha, beta, bound_value)
                samplers.append(m)
            elif type == 'RandomUniform':
                self.__Random_Uniform(params=params, samplers=samplers, random_noise_opt=random_noise_opt)
            elif type == 'RandomNormal':
                self.__Random_Normal(params=params, samplers=samplers, random_noise_opt=random_noise_opt)
            elif type == 'RandomAll':
                r = np.random.randint(3)
                if r == 0:
                    self.__Random_Uniform(params=params, samplers=samplers, random_noise_opt=random_noise_opt)
                elif r == 1:
                    self.__Random_Normal(params=params, samplers=samplers, random_noise_opt=random_noise_opt)
                elif r == 2:
                    self.__Random_Beta(params=params, samplers=samplers, random_noise_opt=random_noise_opt)
            else:
                raise ValueError(f'Do not support distribution: {type} for random noise')
        return samplers, bound_value

    def print_backbone_params(self):
        for key, value in self.net_g.named_parameters():
            logger.info(f'{key}: {value}')

    def _test_parameters(self):
        name = []
        p_now = []
        p_former = []
        for k, v in self.net_g.named_parameters():
            name.append(k)
            p_now.append(v)

        for k, v in self.net_g_former.named_parameters():
            p_former.append(v)

        for i in range(len(name)):
            mean_v = (p_now[i].data - p_former[i].data).abs().mean().item()
            max_v = (p_now[i].data - p_former[i].data).max().item()
            min_v = (p_now[i].data - p_former[i].data).min().item()
            bound = max(abs(max_v), abs(min_v))
            p_name = name[i]
            if bound > 0.0:
                logger.info(f'change of {p_name:<40}: [bound:{bound:6,.4f}] [max:{max_v:6,.4f}] [min:{min_v:6,.4f}] [mean:{mean_v:6,.4f}] ')

    def __Random_Uniform_Elements(self, params, samplers, random_noise_opt):
        bound_value = random_noise_opt['bound_value']
        low = torch.tensor(-bound_value / 2.0, device=torch.device('cuda'))
        high = torch.tensor(bound_value / 2.0, device=torch.device('cuda'))

    def __Random_Uniform(self, params, samplers, random_noise_opt):
        bound_value = random_noise_opt['bound_value']
        bias = bound_value / random_noise_opt['bias_ratio']
        right_sigma = [-1.0, -0.7, -0.5, -0.3, 0.0, 0.3, 0.5]
        left_sigma = [1.0, 0.7, 0.5, 0.3, 0.0, -0.3, -0.5]

        pos_r = np.random.randint(len(right_sigma))
        sigma_r = right_sigma[pos_r]

        pos_l = np.random.randint(len(left_sigma))
        sigma_l = left_sigma[pos_l]

        low = -bound_value / 2.0 + bias * sigma_r
        high = bound_value / 2.0 + bias * sigma_l

        low = torch.full(params.shape, low, device=torch.device('cuda'))
        high = torch.full(params.shape, high, device=torch.device('cuda'))
        bound = torch.full(params.shape, bound_value, device=torch.device('cuda'))

        m = BoundUniform(low=low, high=high, bound_value=bound)

        samplers.append(m)

    def __Random_Normal(self, params, samplers, random_noise_opt):
        bound_value = random_noise_opt['bound_value']
        mean_bound = bound_value / 3
        weight = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        pos = np.random.randint(len(weight))
        r = weight[pos]
        mean = r * mean_bound - mean_bound / 2
        sigma = bound_value / 4
        mean = torch.full(params.shape, mean, device=torch.device('cuda'))
        sigma = torch.full(params.shape, sigma, device=torch.device('cuda'))
        bound = torch.full(params.shape, bound_value, device=torch.device('cuda'))
        m = BoundNormal(mean, sigma, bound)
        samplers.append(m)

    def __Random_Beta(self, params, samplers, random_noise_opt):
        bound_value = random_noise_opt['bound_value']
        pair = [(0.5, 0.5), (0.3, 0.3), (0.7, 0.7), (0.3, 0.5), (0.5, 0.3),
                (0.3, 0.7), (0.7, 0.3), (0.5, 0.7), (0.7, 0.5), (0.5, 0.9),
                (0.9, 0.5), (0.2, 0.9), (0.9, 0.2), (0.1, 0.1), (1.0, 1.0),
                (0.9, 0.9),
                (2.0, 2.0), (4.0, 4.0), (1.0, 3.0), (3.0, 1.0), (1.0, 5.0),
                (5.0, 1.0), (3.0, 2.0), (2.0, 3.0), (2.0, 5.0), (5.0, 2.0)]

        pos = np.random.randint(len(pair))
        alpha = pair[pos][0]
        beta = pair[pos][1]
        alpha = torch.full(params.shape, alpha, device=torch.device('cuda'))
        beta = torch.full(params.shape, beta, device=torch.device('cuda'))
        bound = torch.full(params.shape, bound_value, device=torch.device('cuda'))
        m = BetaDistribution(alpha, beta, bound)
        samplers.append(m)

    def __Random_Value_Beta(self, params, samplers, random_noise_opt):
        ub = random_noise_opt['upper_bound']
        lb = random_noise_opt['lower_bound']
        low = random_noise_opt['low']
        high = random_noise_opt['high'] - low
        alpha = np.random.rand() * high + low
        beta = np.random.rand() * high + low

        alpha = torch.full(params.shape, alpha, device=torch.device('cuda'))
        beta = torch.full(params.shape, beta, device=torch.device('cuda'))

        m = BetaDistribution(alpha, beta, upper_bound=ub, lower_bound=lb)
        samplers.append(m)

    def __Random_Value_Beta2(self, params, samplers, random_noise_opt):
        ub = random_noise_opt['upper_bound']
        lb = random_noise_opt['lower_bound']

        #----------------------------------------------------------------
        low = random_noise_opt['low']
        high = random_noise_opt['high']

        range1 = (low, 1.0 - low)
        range2 = (1.0, high - 1.0)

        pair = [(range1, range1), (range1, range1), (range1, range1),
                (range1, range2), (range2, range1), (range2, range2),
                (range2, range2), (range2, range2)]

        pos = np.random.randint(len(pair))

        alpha_range = pair[pos][0]
        alpha_low = alpha_range[0]
        alpha_high = alpha_range[1]
        alpha = np.random.rand() * alpha_high + alpha_low

        beta_range = pair[pos][1]
        beta_low = beta_range[0]
        beta_high = beta_range[1]
        beta = np.random.rand() * beta_high + beta_low

        alpha = torch.full(params.shape, alpha, device=torch.device('cuda'))
        beta = torch.full(params.shape, beta, device=torch.device('cuda'))

        m = BetaDistribution(alpha, beta, upper_bound=ub, lower_bound=lb)
        samplers.append(m)

    def generate_all_noise(self, range_params, random_noise_opt):
        bound_value = random_noise_opt['bound_value']
        choice = torch.tensor([-bound_value/2.0, -bound_value/4.0, 0.0, bound_value/4.0, bound_value/2.0])
        batch_size = random_noise_opt['batch_size']
        all_noise = []
        for i in range(batch_size):
            batch_noise = []
            for p in range_params:
                pos = torch.randint(len(choice) - 1, p.shape)
                noise = choice[pos]
                batch_noise.append(noise)
            all_noise.append(batch_noise)
        return all_noise
