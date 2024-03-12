import math
import numpy as np
import os
import random
import sys
import time
import torch
import torch.nn as nn
import functools
from pathlib import Path
from torch.distributions.beta import Beta
from os import path as osp
from shutil import get_terminal_size
from torchvision.utils import make_grid
from collections import defaultdict
from torch.distributions.uniform import Uniform

from utils import get_root_logger


def safe_load(file_name, cuda=False):
    try:
        if cuda:
            result = np.load(file_name, allow_pickle=True).item()
        else:
            result = np.load(file_name, allow_pickle=True).item()
        print("sucessfully to load", file_name)
    except:
        print("failed to load", file_name)
        import ipdb; ipdb.set_trace()
        return
    return result

def safe_save(save_path, data):
    # Make sure that the folders exists
    hierarchy = save_path.split("/")
    for i in range(1, len(hierarchy)):
        folder = "/".join(hierarchy[:i])
        if not os.path.exists(folder):
            os.mkdir(folder)

    np.save(save_path, data)
    print("Saved {}".format(save_path))

def print_sparsity(consolidated_masks, percent=1.0, item=False):
    sparsity_dict = {}
    for key in consolidated_masks.keys():
        # Skip output heads
        if "last" in key:
            continue

        mask = consolidated_masks[key]
        if mask is not None:
            sparsity = torch.sum(mask == 1) / np.prod(mask.shape)
            print("{:>12} {:>2.4f}".format(key, sparsity ))

            if item :
                sparsity_dict[key] = sparsity.item() * percent
            else:
                sparsity_dict[key] = sparsity * percent

    return sparsity_dict

def global_sparsity(consolidated_masks):
    denum, num = 0, 0
    for key in consolidated_masks.keys():
        # Skip output heads
        if "last" in key:
            continue

        mask = consolidated_masks[key]
        if mask is not None:
            num += torch.sum(mask == 1).item()
            denum += np.prod(mask.shape)

    return num / denum


def get_between_class_variance(prototypes):
    n = prototypes.size(0)
    dist = torch.pow(prototypes, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist = dist + dist.t()
    dist.addmm_(1, -2, prototypes, prototypes.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    avg_dist = dist.mean()
    return avg_dist

def scandir(dir_path, suffix=None, recursive=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str | obj:`Path`): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.

    Returns:
        A generator for all the interested files with relative pathes.
    """
    if isinstance(dir_path, (str, Path)):
        dir_path = str(dir_path)
    else:
        raise TypeError('"dir_path" must be a string or Path object')

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                rel_path = osp.relpath(entry.path, root)
                if suffix is None:
                    yield rel_path
                elif rel_path.endswith(suffix):
                    yield rel_path
            else:
                if recursive:
                    yield from _scandir(
                        entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)

def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)

def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())

def check_resume(opt, resume_iter):
    """Check resume states and pretrain_model paths.

    Args:
        opt (dict): Options.
        resume_iter (int): Resume iteration.
    """
    logger = get_root_logger()
    if opt['path']['resume_state']:
        # ignore pretrained model paths
        if opt['path'].get('pretrain_model_g') is not None or opt['path'].get(
                'pretrain_model_d') is not None:
            logger.warning(
                'pretrain_model path will be ignored during resuming.')

        # set pretrained model paths
        opt['path']['pretrain_model_g'] = osp.join(opt['path']['models'],
                                                   f'net_g_{resume_iter}.pth')
        logger.info(
            f"Set pretrain_model_g to {opt['path']['pretrain_model_g']}")

        opt['path']['pretrain_model_d'] = osp.join(opt['path']['models'],
                                                   f'net_d_{resume_iter}.pth')
        logger.info(
            f"Set pretrain_model_d to {opt['path']['pretrain_model_d']}")

def mkdir_and_rename(path):
    """mkdirs. If path exists, rename it with timestamp and create a new one.

    Args:
        path (str): Folder path.
    """
    if osp.exists(path):
        new_name = path + '_archived_' + get_time_str()
        print(f'Path already exists. Rename it to {new_name}', flush=True)
        os.rename(path, new_name)
    mkdir_or_exist(path)

def make_exp_dirs(opt):
    """Make dirs for experiments."""
    path_opt = opt['path'].copy()
    if opt['is_train']:
        mkdir_and_rename(path_opt.pop('experiments_root'))
    elif opt['is_incremental']:
        mkdir_and_rename(path_opt.pop('incremental_root'))
    else:
        mkdir_and_rename(path_opt.pop('results_root'))
    path_opt.pop('strict_load')
    for key, path in path_opt.items():
        if 'pretrain' not in key and 'resume' not in key and 'base' not in key and 'index' not in key:
            mkdir_or_exist(path)

def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_gpu(gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    print('using gpu:', gpu)

def crop_border(imgs, crop_border):
    """Crop borders of images.

    Args:
        imgs (list[ndarray] | ndarray): Images with shape (h, w, c).
        crop_border (int): Crop border for each end of height and weight.

    Returns:
        list[ndarray]: Cropped images.
    """
    if crop_border == 0:
        return imgs
    else:
        if isinstance(imgs, list):
            return [
                v[crop_border:-crop_border, crop_border:-crop_border, ...]
                for v in imgs
            ]
        else:
            return imgs[crop_border:-crop_border, crop_border:-crop_border,
                        ...]

def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list)
             and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(
            f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(
                _tensor, nrow=int(math.sqrt(_tensor.size(0))),
                normalize=False).numpy()
            img_np = np.transpose(img_np[[2, 1, 0], :, :],
                                  (1, 2, 0))  # HWC, BGR
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = np.transpose(img_np[[2, 1, 0], :, :],
                                  (1, 2, 0))  # HWC, BGR
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError('Only support 4D, 3D or 2D tensor. '
                            f'But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result

def dir_size(path):
    total_size = 0
    file_num = 0
    for dir in os.listdir(path):
        sub_path = osp.join(path, dir)
        if os.path.isfile(sub_path):
            file_num += 1
            total_size += os.path.getsize(sub_path)
        elif os.path.isdir(sub_path):
            sz, fn = dir_size(sub_path)
            total_size += sz
            file_num += fn
    return total_size, file_num

class ProgressBar(object):
    """A progress bar that can print the progress.

    Modified from:
    https://github.com/hellock/cvbase/blob/master/cvbase/progress.py
    """

    def __init__(self, task_num=0, bar_width=50, start=True):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = (
            bar_width if bar_width <= max_bar_width else max_bar_width)
        self.completed = 0
        if start:
            self.start()

    def _get_max_bar_width(self):
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            print(f'terminal width is too small ({terminal_width}), '
                  'please consider widen the terminal for better '
                  'progressbar visualization')
            max_bar_width = 10
        return max_bar_width

    def start(self):
        if self.task_num > 0:
            sys.stdout.write(f"[{' ' * self.bar_width}] 0/{self.task_num}, "
                             f'elapsed: 0s, ETA:\nStart...\n')
        else:
            sys.stdout.write('completed: 0, elapsed: 0s')
        sys.stdout.flush()
        self.start_time = time.time()

    def update(self, msg='In progress...'):
        self.completed += 1
        elapsed = time.time() - self.start_time + 1e-8
        fps = self.completed / elapsed
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            mark_width = int(self.bar_width * percentage)
            bar_chars = '>' * mark_width + '-' * (self.bar_width - mark_width)
            sys.stdout.write('\033[2F')  # cursor up 2 lines
            sys.stdout.write(
                '\033[J'
            )  # clean the output (remove extra chars since last display)
            sys.stdout.write(
                f'[{bar_chars}] {self.completed}/{self.task_num}, '
                f'{fps:.1f} task/s, elapsed: {int(elapsed + 0.5)}s, '
                f'ETA: {eta:5}s\n{msg}\n')
        else:
            sys.stdout.write(
                f'completed: {self.completed}, elapsed: {int(elapsed + 0.5)}s,'
                f' {fps:.1f} tasks/s')
        sys.stdout.flush()

def sample_data(data, labels, way, shots):
    batch = []
    if isinstance(labels, list):
        class_labels = labels[0]
    else:
        class_labels = labels

    label_list = torch.unique(class_labels)
    index_dic = defaultdict(list)

    for index, label in enumerate(class_labels):
        index_dic[label.item()].append(index)

    for key in index_dic.keys():
        index_dic[key] = torch.tensor(index_dic[key])

    classes_set = torch.randperm(len(label_list))[:way]
    for cl in classes_set:
        class_label = label_list[cl]
        index_list = index_dic[class_label.item()]
        pos = torch.randperm(len(index_list))[:shots].tolist()
        batch.append(index_list[pos])
    batch = torch.stack(batch).reshape(-1)

    if isinstance(labels, list):
        labels = [lb[batch] for lb in labels]
        return data[batch], labels
    else:
        return data[batch], labels[batch]

class Averager():

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.data = []

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def add(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.data.append(val)

    def item(self):
        return self.avg

    def obtain_data(self):
        return self.data

    def __len__(self):
        return len(self.data)

def one_hot(y, num_class):
    return torch.zeros((len(y), num_class)).cuda().scatter_(1, y.unsqueeze(1), 1)

class Timer():
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return f'{x / 3600:.1f}h'
        if x >= 60:
            return f'{round(x / 60)}m'
        return f'{x}s'

class DiscreteUniform():
    def __init__(self, bound, shape, reduction_factor):
        self.bound = bound
        self.shape = shape
        b = int(reduction_factor / 2)
        ratio = torch.arange(-b, b+1, device=torch.device('cuda'))
        self.choice = ratio / reduction_factor
        self.choice = self.choice * bound
        self.logger = get_root_logger()
        # for r in self.choice:
        #     self.logger.info(f'{r}')


    def sample(self):
        pos = self.generate_pos()
        noise = self.choice[pos]
        # for r in self.choice:
        #     sum = (noise == r).sum()
        #     self.logger.info(f'{r}: {sum/len(noise)}')
        return noise

    def generate_pos(self):
        return torch.randint(len(self.choice), self.shape, device=torch.device('cuda'))

class DiscreteUniform2():
    def __init__(self, bound, shape, reduction_factor):
        self.bound = bound
        self.shape = shape
        ratio1 = torch.tensor([-2, -2, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2], device=torch.device('cuda'))
        ratio2 = torch.tensor([-2, -2, -2, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2], device=torch.device('cuda'))
        self.choice = ratio1 / 4
        self.choice = self.choice * bound
        self.logger = get_root_logger()
        # for r in self.choice:
        #     self.logger.info(f'{r}')

    def sample(self):
        pos = self.generate_pos()
        noise = self.choice[pos]
        # str = ''
        # unique = torch.unique_consecutive(self.choice)
        # for r in unique:
        #     sum = (noise == r).sum()
        #     str += f'!!! {sum/len(noise):.5f} '
        # self.logger.info(str)
        return noise

    def generate_pos(self):
        return torch.randint(len(self.choice), self.shape, device=torch.device('cuda'))

class DiscreteUniform3():
    def __init__(self, bound, shape, reduction_factor):
        self.bound = bound
        self.shape = shape
        ratio1 = torch.tensor([-2, -2, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2], device=torch.device('cuda'))

        self.choice = ratio1 / 4
        self.choice = self.choice * bound
        self.logger = get_root_logger()
        # for r in self.choice:
        #     self.logger.info(f'{r}')

    def sample(self):
        pos = self.generate_pos()
        noise = self.choice[pos]
        # str = ''
        # unique = torch.unique_consecutive(self.choice)
        # for r in unique:
        #     sum = (noise == r).sum()
        #     str += f'!!! {sum/len(noise):.5f} '
        # self.logger.info(str)
        return noise

    def generate_pos(self):
        return torch.randint(len(self.choice), self.shape, device=torch.device('cuda'))

class BetaDistribution():
    def __init__(self, alpha, beta, upper_bound, lower_bound):
        self.m = Beta(alpha, beta)
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.length = self.upper_bound - self.lower_bound

    def sample(self):
        ans = self.m.sample()
        ans = ans * self.length + self.lower_bound
        return ans

class DiscreteBetaDistribution():
    def __init__(self, low, high, shape, bound, reduction_factor):
        self.low = low
        self.high = high
        self.shape = shape

        self.bound = bound
        b = int(reduction_factor / 2)
        ratio = torch.arange(-b, b + 1, device=torch.device('cuda'))
        self.choice = ratio / reduction_factor
        self.anchors = torch.arange(self.choice.shape[0]+1) / (self.choice.shape[0])
        self.choice = self.choice * bound

        self.alpha = torch.tensor([0.3, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0], device=torch.device('cuda'))
        self.beta = torch.tensor([0.3, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0], device=torch.device('cuda'))

    def sample(self):
        a_p = np.random.randint(7)
        b_p = np.random.randint(7)
        alpha = self.alpha[a_p]
        beta = self.beta[b_p]
        alpha = torch.full(self.shape, alpha, device=torch.device('cuda'))
        beta = torch.full(self.shape, beta, device=torch.device('cuda'))

        m = Beta(alpha, beta)

        ans = m.sample()

        for i in range(self.anchors.shape[0]-1):
            a1 = self.anchors[i]
            a2 = self.anchors[i+1]
            p = torch.logical_and(ans >= a1, ans <= a2)
            ans[p] = i
        ans = ans.long()
        ans = self.choice[ans]

        return ans

class BoundNormal():
    def __init__(self, mean, sigma, bound_value):
        self.m = torch.distributions.normal.Normal(mean, sigma)
        self.mean = mean
        self.sigma = sigma
        self.bound_value = bound_value

    def sample(self):
        ans = self.m.sample()
        upper_bound = self.bound_value / 2
        lower_bound = - self.bound_value / 2
        upper_mask = torch.gt(ans, upper_bound)
        lower_mask = torch.lt(ans, lower_bound)

        # n_upper_pos = upper_mask.sum()
        # n_lower_pos = lower_mask.sum()
        # n_total_pos = torch.ones(ans.shape).sum()
        #
        # if n_lower_pos > 0 or n_lower_pos < 0:
        #     logger = get_root_logger()
        #     logger.debug(f'mean:{self.mean}, sigma:{self.sigma}, n_upper_pos: {n_upper_pos}, n_lower_pos: {n_lower_pos}, total_pos: {n_total_pos}')

        ans[upper_mask] = upper_bound[upper_mask]
        ans[lower_mask] = lower_bound[lower_mask]

        return ans

class BoundUniform():
    def __init__(self, low, high, bound_value):
        self.m = Uniform(low, high)
        self.low = low
        self.high = high
        self.bound_value = bound_value

    def sample(self):
        ans = self.m.sample()
        upper_bound = self.bound_value / 2
        lower_bound = - self.bound_value / 2
        upper_mask = torch.gt(ans, upper_bound)
        lower_mask = torch.lt(ans, lower_bound)

        # n_upper_pos = upper_mask.sum()
        # n_lower_pos = lower_mask.sum()
        # n_total_pos = torch.ones(ans.shape).sum()
        #
        # if n_lower_pos > 0 or n_lower_pos < 0:
        #     logger = get_root_logger()
        #     logger.debug(f'low:{self.low}, high:{self.high}, n_upper_pos: {n_upper_pos}, n_lower_pos: {n_lower_pos}, total_pos: {n_total_pos}')

        ans[upper_mask] = upper_bound[upper_mask]
        ans[lower_mask] = lower_bound[lower_mask]

        return ans

class UniformElements():
    def __init__(self, low, high, params_shape):
        self.m = Uniform(low, high)
        self.low = low
        self.high = high
        self.params_shape = params_shape

    def sample(self):
        # TODO finish element-wise Uniform distribution
        return

def pnorm(data, p):
    normB = torch.norm(data, 2, dim=1)
    for i in range(data.size(0)):
        data[i] = data[i] / torch.pow(normB[i], p)
    return data

class AvgDict():
    def __init__(self):
        self.dict = {}

    def add_dict(self, d):
        for key, value in d.items():
            if self.dict.get(key, None) is None:
                self.dict[key] = Averager()
                self.dict[key].add(value)
            else:
                self.dict[key].add(value)

    def get_ordinary_dict(self):
        d = {key: value.item() for key, value in self.dict.items()}
        return d


