# coding=utf-8
from __future__ import absolute_import, print_function
import argparse
import datetime
import logging
import math
import random
import os
import time
import torch
from utils import get_time_str
from os import path as osp

import numpy as np
from copy import deepcopy, copy

from data import create_dataloader, create_dataset, create_sampler
from methods import create_model
from utils.options import dict2str, parse
from utils import (MessageLogger, get_env_info, get_root_logger,
                   init_tb_logger, init_wandb_logger, check_resume,
                   make_exp_dirs, set_random_seed, set_gpu, Averager)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt',type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = parse(args.opt, is_train=False, is_incremental=True)

    rank = 0
    world_size = 1
    opt['rank'] = 0
    opt['world_size'] = 1

    make_exp_dirs(opt)

    log_file = osp.join(opt['path']['log'],
                        f"incremental_{opt['name']}_{get_time_str()}.log")

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

    # set gpu
    # set_gpu(opt['gpu'])

    # set random seed
    seed = opt['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    logger.info(f'Random seed: {seed}')
    set_random_seed(seed + rank)

    torch.backends.cudnn.benchmark = True


    # define the variables for incremental few-shot learning
    total_classes = opt['datasets']['train']['total_classes']
    bases = opt['train']['bases']
    num_tasks = opt['train']['tasks']
    num_shots = opt['train']['shots']
    fine_tune = opt['train']['fine_tune']
    fine_tune_epoch = opt['train']['fine_tune_epoch']
    num_class_per_task = int((total_classes - bases) / (num_tasks - 1))
    opt['train']['num_class_per_task'] = num_class_per_task

    if opt.get('Random', True):
        random_class_perm = np.random.permutation(total_classes)
    else:
        random_class_perm = np.arange(total_classes)

    opt['class_permutation'] = random_class_perm

    # deep copy the opt
    opt_old = deepcopy(opt)

    num_tests = opt['train']['num_test']
    acc_avg = [Averager() for i in range(num_tasks)]
    for test_id in range(num_tests):
        try:
            opt = deepcopy(opt_old)
        except:
            opt = copy(opt_old)
        for task_id in range(num_tasks):
            opt['test_id'] = test_id
            # deep copy the opt
            # Load the model of former session
            # 'task_id = -1' indicates that the program will not load the prototypes, and just load the base model
            opt['task_id'] = task_id - 1

            model = create_model(opt)

            opt['task_id'] = task_id

            val_classes = random_class_perm[:bases + task_id * num_class_per_task]
            if task_id == 0:
                selected_classes = random_class_perm[:bases]
            else:
                selected_classes = random_class_perm[bases + (task_id - 1) * num_class_per_task:
                                                     bases + task_id * num_class_per_task]


            IL = opt['train'].get('IL', False)

            # creating the dataset
            for phase, dataset_opt in opt['datasets'].items():
                if phase == 'train':
                    dataset_opt['all_classes'] = random_class_perm
                    dataset_opt['selected_classes'] = selected_classes
                    train_set = create_dataset(dataset_opt=dataset_opt)
                    if task_id > 0 and not IL:
                        session_path_root, _ = os.path.split(dataset_opt['dataroot'])
                        Random = opt.get('Random', True)
                        if opt['manual_seed'] != 1997:
                            seed = opt['manual_seed']
                            index_root = osp.join(session_path_root,
                                                  f'Random{Random}_seed{seed}_bases{bases}_tasks{num_tasks}_shots{num_shots}',
                                                  f'test_{test_id}', f'session_{task_id}', 'index.pt')
                        else:
                            index_root = osp.join(session_path_root,
                                                  f'Random{Random}_seed{seed}_bases{bases}_tasks{num_tasks}_shots{num_shots}',
                                                  f'test_{test_id}', f'session_{task_id}', 'index.pt')

                        index = torch.load(index_root)
                        train_set.sample_the_buffer_data_with_index(index)

                if phase == 'val':
                    dataset_opt['all_classes'] = random_class_perm
                    dataset_opt['selected_classes'] = val_classes
                    val_set = create_dataset(dataset_opt=dataset_opt)

            model.incremental_init(train_set, val_set)

            if task_id > 0 and fine_tune:
                tb_logger_temp = tb_logger if test_id == 0 else None
                model.incremental_fine_tune(train_dataset=train_set, val_dataset=val_set,
                                                 num_epoch=fine_tune_epoch, task_id=task_id, test_id=test_id,
                                                 tb_logger=tb_logger_temp)
                print('fine-tune procedure is finished!')

            # update incremental setting
            model.incremental_update(train_set)

            acc = model.incremental_test(val_set, task_id, test_id)
            acc_avg[task_id].add(acc)

            model.save(epoch=-1, current_iter=task_id, name=f'test{test_id}_session', dataset=train_set)
            opt = deepcopy(opt_old)
            model.set_the_saving_files_path(opt=opt, task_id=task_id)
            print(f'Successfully saving the model of test {test_id} session {task_id}')

    message = f'--------------------------Final Avg Acc-------------------------'
    logger.info(message)

    for i, acc in enumerate(acc_avg):
        data = acc.obtain_data()
        m = np.mean(data)
        std = np.std(data)
        pm = 1.96 * (std / np.sqrt(len(data)))

        message = f'Session {i + 1}: {m*100:.2f}+-{pm*100:.2f}'
        logger.info(message)
        if tb_logger:
            tb_logger.add_scalar(f'sessions_acc', acc.item(), i)
            if wandb_logger is not None:
                wandb_logger.log({f'sessions_acc': acc.item()}, step=i)

    print('finish!!')

    print('finish')

if __name__ == '__main__':
    main()
