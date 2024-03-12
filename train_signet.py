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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt',type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('--local_rank', type=int, default=0)
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
    for phase, dataset_opt in opt['datasets'].items():
        dataset_opt['all_classes'] = random_class_perm
        dataset_opt['selected_classes'] = selected_classes
        sampler_opt = dataset_opt['sampler']
        if sampler_opt.get('num_classes', None) is None:
            sampler_opt['num_classes'] = num_classes

        if phase == 'train':
            dataset_opt['batch_size'] = dataset_opt['batch_size_base_classes']
            dataset_opt['task_id'] = 0
            train_set = create_dataset(dataset_opt)
            opt['train_set'] = train_set
            train_sampler = create_sampler(train_set, sampler_opt)

            train_loader = create_dataloader(
                train_set,
                dataset_opt,
                sampler=train_sampler,
                seed=seed)

            logger.info(
                'Training statistics:'
                f'\n\tNumber of train classes: {num_classes}'
                f'\n\tBatch size: {dataset_opt["batch_size"]}'
                f'\n\tTotal epochs: {opt["train"]["epoch"]}')

        elif phase == 'val':
            dataset_opt['task_id'] = 0
            val_set = create_dataset(dataset_opt)
            val_set.set_aug(False)
            val_loader = create_dataloader(
                val_set,
                dataset_opt,
                sampler=None,
                seed=seed)
            logger.info(
                f'Number of val images/folders in {dataset_opt["name"]}: '
                f'{len(val_set)}')
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')
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
    max_acc, acc = 0.0, 0.0
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

            sparsity=True if epoch > opt['train']['s_epoch'] else None # miniImageNet
            if True:
                model.optimize_parameters(current_iter, mask=per_task_masks[0],
                                          sparsity=sparsity)
            else:
                ''' try to learn f2m'''
                model.optimize_f2m_parameters(current_iter,
                                              mask=per_task_masks[0],
                                              sparsity=sparsity)

            # get model masks
            per_task_masks[0] = model.get_masks(0)

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
            if opt['val']['val_freq'] is not None and current_iter % opt['val']['val_freq'] == 0:
                train_set.set_aug(False)
                mask = per_task_masks[0]
                if mask is None:
                    mask = model.get_masks(0)
                acc = model.validation(train_set, val_loader,
                                       current_iter, tb_logger,
                                       mask=mask)
                if acc > max_acc:
                    max_acc = acc
                    model.save(epoch, -1, name='best_net',
                               mask=per_task_masks, task_id=0)

                    model.save_mask(per_task_masks, task_id=0,
                                    net_label='best_net',
                                    current_iter=-1)

                train_set.set_aug(True)

        logger.info(f'ETA:{timer.measure()}/{timer.measure((epoch + 1)/ total_epoch)}')

    # end of epoch
    logger.info('Save the latest model if the best')

    train_set.set_aug(False)
    acc = model.validation(train_set, val_loader,
                           current_iter, tb_logger,
                           mask=per_task_masks[0])
    if acc > max_acc:
        model.save(epoch, -1, name='best_net',
                   mask=per_task_masks, task_id=0)
        model.save_mask(per_task_masks, task_id=0, net_label='best_net',
                        current_iter=-1)

    logger.info(f'Best acc is {max_acc:.4f}')
    if opt['val']['val_freq'] is not None:
        model.validation(train_set, val_loader, current_iter, tb_logger, mask=per_task_masks[0])

    if tb_logger is not None:
        tb_logger.close()
    if wandb_logger is not None:
        wandb.finish()


if __name__ == '__main__':

    main()

