# coding=utf-8
from __future__ import absolute_import, print_function
import argparse
import datetime
import logging
import math
import random
import time

import torch
import wandb

from os import path as osp
import numpy as np

from data import create_dataloader, create_dataset, create_sampler
from methods import create_model
from utils.options import dict2str, parse
from utils import (MessageLogger, get_env_info, get_root_logger,
                   init_tb_logger, init_wandb_logger, check_resume,
                   make_exp_dirs, set_random_seed, get_time_str, Timer)

from dataloader.data_utils import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt',type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('-dataset', type=str, default='cub200',
                        choices=['mini_imagenet', 'cub200', 'cifar100'])
    parser.add_argument('-dataroot', type=str, default='DataSet/')

    parser.add_argument('--base_class', type=int, default=100)
    parser.add_argument('--num_classes', type=int, default=200)
    parser.add_argument('--way', type=int, default=10)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--sessions', type=int, default=11)
    parser.add_argument('--batch_size_base', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--autoaug', type=int, default=0) # 1
    parser.add_argument('--train', type=bool, default=True)

    args = parser.parse_args()
    opt = parse(args.opt, is_train=True)

    rank = 0
    opt['rank'] = 0
    opt['world_size'] = 1

    # load resume states if exists
    if opt['path'].get('resume_state'):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt['path']['resume_state'],
            map_location=lambda storage, loc: storage.cuda(device_id))
    else:
        resume_state = None

    # mkdir and loggers
    if resume_state is None:
        make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'],
                        f"train_{opt['name']}_{get_time_str()}.log")

    logger = get_root_logger(
        logger_name='FS-IL', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # initialize tensorboard logger and wandb logger
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        log_dir = './tb_logger_{}/'.format(opt['datasets']['train']['name']) + opt['name']
        tb_logger = init_tb_logger(log_dir=log_dir)


    if (opt['logger'].get('wandb')
        is not None) and (opt['logger']['wandb'].get('project')
                          is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, (
            'should turn on tensorboard when using wandb')
        wandb_logger = init_wandb_logger(opt)
    else:
        wandb_logger = None
    opt['wandb_logger'] = wandb_logger

    # set random seed
    seed = opt['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    logger.info(f'Random seed: {seed}')
    set_random_seed(seed + rank)
    torch.set_num_threads(1)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # calculate the number of tasks for each new task
    bases = opt['train']['bases']
    total_classes = opt['datasets']['train']['total_classes']

    if opt.get('Random', True):
        random_class_perm = np.random.permutation(total_classes)
    else:
        random_class_perm = np.arange(total_classes)
        # randomly generate the sorting of categories

    num_classes = bases
    # select the classes for training
    selected_classes = random_class_perm[:bases]

    # create train and val dataloaders
    train_loader, val_loader = None, None

    # split-FSLL
    args = set_up_datasets(args)
    args.train=False
    train_set_val, train_loader_val, val_set, val_loader = get_base_dataloader(args)

    args.train=True
    #train_set, train_loader, val_set, val_loader = get_base_dataloader(args)
    train_set, train_loader, _, _ = get_base_dataloader(args)

    assert train_loader is not None

    # create model
    if resume_state:
        check_resume(opt, resume_state['iter'])  # modify pretrain_model paths

    model = create_model(opt)

    # TODO resume training
    if resume_state:
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, "
                    f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        start_epoch = 0
        current_iter = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger, wandb_logger)

    # training
    logger.info(
        f'Start training from epoch: {start_epoch}, iter: {current_iter}')

    total_epoch = opt['train']['epoch']
    max_acc = 0.0
    timer = Timer()
    model.init_training(train_set)

    per_task_masks, consolidated_masks = {}, {}
    per_task_masks[0] = None
    for epoch in range(start_epoch, total_epoch + 1):
        if epoch == 0 :
            pass

        for i, data in enumerate(train_loader, 0):
            current_iter += 1

            # update learning rate
            model.update_learning_rate(
                current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))

            # training
            model.feed_data(data)
            smooth=True if epoch > opt['train']['s_epoch'] else None # miniImageNet
            if opt['subnet_type'] == 'softnet':
                smooth=True if epoch > opt['train']['s_epoch'] else False
            else:
                smooth=False

            model.optimize_parameters(current_iter,
                                      mask=per_task_masks[0],
                                      smooth=smooth)

            # get model masks
            per_task_masks[0] = model.get_masks(-1)

            # log
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            # save models and training states
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter, mask=per_task_masks, task_id=0)

            # validation
            if opt['val']['val_freq'] is not None and current_iter % opt[
                'val']['val_freq'] == 0:
                mask = per_task_masks[0]
                if mask is None:
                    mask = model.get_masks(-1)
                acc = model.validation(train_set_val, val_loader,
                                       current_iter, tb_logger,
                                       mask=mask)
                if acc > max_acc:
                    max_acc = acc
                    model.save(epoch, -1, name='best_net',
                               mask=per_task_masks, task_id=0)

        logger.info(f'ETA:{timer.measure()}/{timer.measure((epoch + 1)/ total_epoch)}')

    # end of epoch
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    logger.info(f'Best acc is {max_acc:.4f}')
    if opt['val']['val_freq'] is not None:
        model.validation(train_set_val, val_loader, current_iter, tb_logger, mask=per_task_macsks[0])

    if tb_logger is not None:
        tb_logger.close()
    if wandb_logger is not None:
        wandb.finish()


if __name__ == '__main__':

    main()

