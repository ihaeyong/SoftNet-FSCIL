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
import os

from os import path as osp
from copy import deepcopy
import numpy as np

from data import create_dataloader, create_dataset, create_sampler
from methods import create_model
from utils.options import dict2str, parse
from utils import (MessageLogger, get_env_info, get_root_logger,
                   init_tb_logger, init_wandb_logger, check_resume,
                   make_exp_dirs, set_random_seed, get_time_str, Timer)


def generate_training_dataset(opt, task_id):
    random_class_perm = opt['class_permutation']
    total_classes = opt['datasets']['train']['total_classes']
    bases = opt['train']['bases']
    Random = opt['Random']
    seed = opt['manual_seed']
    num_tasks = opt['train']['tasks']
    num_shots = opt['train']['shots']
    num_class_per_task = int((total_classes - bases) / (num_tasks - 1))

    dataset_opt = opt['datasets']['train']
    dataset_opt['all_classes'] = random_class_perm

    base_classes = random_class_perm[:bases]

    dataset_opt['selected_classes'] = base_classes
    train_set_bases = create_dataset(dataset_opt)

    for i in range(1, task_id+1):
        selected_classes = random_class_perm[bases + (i - 1) * num_class_per_task:
                                             bases + i * num_class_per_task]

        dataset_opt['selected_classes'] = selected_classes
        train_set_novel = create_dataset(dataset_opt)

        session_path_root, _ = os.path.split(dataset_opt['dataroot'])
        index_root = osp.join(session_path_root,
                              f'Random{Random}_seed{seed}_bases{bases}_tasks{num_tasks}_shots{num_shots}',
                              f'test_{3}', f'session_{i}', 'index.pt')

        index = torch.load(index_root)
        train_set_novel.sample_the_buffer_data_with_index(index)
        train_set_bases.combine_another_dataset(train_set_novel)

    sampler_opt = dataset_opt['sampler']
    if sampler_opt.get('num_classes', None) is None:
        sampler_opt['num_classes'] = opt['network_g']['num_classes']

    train_sampler = create_sampler(train_set_bases, sampler_opt)

    train_loader = create_dataloader(
        train_set_bases,
        dataset_opt,
        sampler=train_sampler,
        seed=opt['manual_seed'])

    return train_set_bases, train_loader


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
        log_dir = './tb_logger/' + opt['name']
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
    total_classes = opt['datasets']['train']['total_classes']
    bases = opt['train']['bases']
    num_tasks = opt['train']['tasks']
    task_id = opt['train']['task_id']
    num_class_per_task = int((total_classes - bases) / (num_tasks - 1))
    opt['train']['num_class_per_task'] = num_class_per_task

    # randomly generate the sorting of categories
    if opt.get('Random', True):
        random_class_perm = np.random.permutation(total_classes)
    else:
        random_class_perm = np.arange(total_classes)
        # randomly generate the sorting of categories

    opt['class_permutation'] = random_class_perm

    n_novel_classes = task_id * num_class_per_task
    opt['network_g']['num_classes'] = bases + n_novel_classes

    # create train and val dataloaders
    train_loader, val_loader = None, None

    val_classes = random_class_perm[:bases + task_id * num_class_per_task]

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set_bases, train_loader = generate_training_dataset(opt, task_id)
            # set the milestones
            milestones = opt['train']['scheduler']['milestones']
            n_batch = int(len(train_loader.sampler) / dataset_opt['batch_size'])
            opt['train']['scheduler']['milestones'] = [m * n_batch for m in milestones]

        elif phase == 'val':
            dataset_opt['all_classes'] = random_class_perm
            dataset_opt['selected_classes'] = val_classes
            val_set = create_dataset(dataset_opt)
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

    # create the model
    model = create_model(opt)

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

    for epoch in range(start_epoch, total_epoch + 1):
        for i, data in enumerate(train_loader, 0):

            current_iter += 1

            # update learning rate
            model.update_learning_rate(
                current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))

            # training
            model.feed_data(data)
            model.optimize_parameters(current_iter)

            # log
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            # save models and training states
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)

            # validation
            if opt['val']['val_freq'] is not None and current_iter % opt[
                'val']['val_freq'] == 0:
                train_set_bases.set_aug(False)
                acc = model.validation(train_set_bases, val_loader, current_iter, tb_logger)
                if acc > max_acc:
                    max_acc = acc
                    model.save(epoch, -1, name='best_net')
                train_set_bases.set_aug(True)

        logger.info(f'ETA:{timer.measure()}/{timer.measure((epoch + 1) / total_epoch)}')
        if epoch == total_epoch:
            acc = model.validation(train_set_bases, val_loader, current_iter, tb_logger)
            logger.info(f'The latest acc is {acc:.4f}')

    # end of epoch
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    logger.info(f'Best acc is {max_acc:.4f}')

    # if opt['val']['val_freq'] is not None:
    #     model.validation(train_set_bases, val_loader, current_iter, tb_logger)

    if tb_logger is not None:
        tb_logger.close()
    if wandb_logger is not None:
        wandb.finish()


if __name__ == '__main__':

    main()

