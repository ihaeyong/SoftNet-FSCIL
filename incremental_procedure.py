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

def generate_training_dataset(opt, task_id, test_id):
    random_class_perm = opt['class_permutation']
    total_classes = opt['datasets']['train']['total_classes']
    Random = opt['Random']
    seed = opt['manual_seed']
    bases = opt['train']['bases']
    num_tasks = opt['train']['tasks']
    num_shots = opt['train']['shots']
    num_class_per_task = int((total_classes - bases) / (num_tasks - 1))

    dataset_opt = opt['datasets']['train']
    dataset_opt['all_classes'] = random_class_perm

    train_set = None
    if opt['train']['novel_exemplars'] > 0:
        for i in range(1, task_id+1):
            dataset_opt['task_id'] = i
            selected_classes = random_class_perm[bases + (i - 1) * num_class_per_task:
                                                 bases + i * num_class_per_task]

            dataset_opt['selected_classes'] = selected_classes
            train_set_novel = create_dataset(dataset_opt)

            session_path_root, _ = os.path.split(dataset_opt['dataroot'])

            index_root = osp.join(session_path_root,
                                  f'Random{Random}_seed{seed}_bases{bases}_tasks{num_tasks}_shots{num_shots}',
                                  f'test_{test_id}', f'session_{i}', 'index.pt')

            index = torch.load(index_root)
            train_set_novel.sample_the_buffer_data_with_index(index)
            if i < task_id:
                train_set_novel.sample_the_buffer_data(opt['train']['novel_exemplars'])

            if train_set is not None:
                train_set.combine_another_dataset(train_set_novel)
            else:
                train_set = train_set_novel
    else:
        selected_classes = random_class_perm[bases + (task_id-1) * num_class_per_task:
                                             bases + task_id * num_class_per_task]

        dataset_opt['selected_classes'] = selected_classes
        train_set = create_dataset(dataset_opt)

        session_path_root, _ = os.path.split(dataset_opt['dataroot'])
        index_root = osp.join(session_path_root,
                              f'bases{bases}_tasks{num_tasks}_shots{num_shots}',
                              f'test_{test_id}', f'session_{task_id}', 'index.pt')
        index = torch.load(index_root)
        train_set.sample_the_buffer_data_with_index(index)


    sampler_opt = dataset_opt['sampler']
    if sampler_opt.get('num_classes', None) is None:
        sampler_opt['num_classes'] = task_id * num_class_per_task

    dataset_opt['batch_size'] = len(train_set)
    # dataset_opt['batch_size'] = 100

    train_sampler = create_sampler(train_set, sampler_opt)

    train_loader = create_dataloader(
        train_set,
        dataset_opt,
        sampler=train_sampler,
        seed=opt['manual_seed'])

    return train_set, train_loader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt',type=str, required=True, help='Path to option YAML file.')
    args = parser.parse_args()
    opt = parse(args.opt, is_train=False, is_incremental=True)

    rank = 0
    opt['rank'] = 0
    opt['world_size'] = 1

    opt['name'] += '_' + opt['subnet_type'] + str(1.0 - opt['network_g']['sparsity'])
    g_opt_name = opt['name']

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

    # define the variables for incremental few-shot learning
    total_classes = opt['datasets']['train']['total_classes']
    bases = opt['train']['bases']
    num_tasks = opt['train']['tasks']
    num_shots = opt['train']['shots']

    fine_tune_epoch = opt['train'].get('fine_tune_epoch', None)

    num_class_per_task = int((total_classes - bases) / (num_tasks - 1))
    opt['train']['num_class_per_task'] = num_class_per_task

    if opt.get('Random', True):
        random_class_perm = np.random.permutation(total_classes)
    else:
        random_class_perm = np.arange(total_classes)
        # randomly generate the sorting of categories

    opt['class_permutation'] = random_class_perm

    # deep copy the opt
    opt_old = deepcopy(opt)

    # Test the session 1 and save the prototypes
    opt['task_id'] = -1
    opt['test_id'] = 0
    model = create_model(opt)
    opt['task_id'] = 0

    val_classes = random_class_perm[:bases]
    selected_classes = random_class_perm[:bases]

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_opt['task_id'] = 0
            dataset_opt['all_classes'] = random_class_perm
            dataset_opt['selected_classes'] = selected_classes
            train_set = create_dataset(dataset_opt=dataset_opt, info=False)

        if phase == 'val':
            dataset_opt['task_id'] = 0
            dataset_opt['all_classes'] = random_class_perm
            dataset_opt['selected_classes'] = val_classes
            val_set = create_dataset(dataset_opt=dataset_opt, info=False)

    model.incremental_init(train_set, val_set)
    if opt['path'].get('pretrain_prototypes', None) is None:
        model.incremental_update(novel_dataset=train_set)
    if opt.get('Test1', True):
        if opt.get('details', False):
            acc, acc_former_ave, acc_former_all_ave, acc_novel_all_ave = model.incremental_test(val_set, 0, 0)
        else:
            acc = model.incremental_test(val_set, 0, 0)
    if opt['path'].get('pretrain_prototypes', None) is None:
        pt_path, _ = os.path.split(opt['path']['base_model'])
        pt_path = osp.join(pt_path, 'pretrain_prototypes.pt')
        torch.save(model.prototypes_dict, pt_path)
    model.save(epoch=-1, current_iter=0, name=f'test{0}_session', dataset=train_set)

    num_tests = opt['train']['num_test']
    acc_avg = [Averager() for i in range(num_tasks)]
    acc_former_ave_avg = [Averager() for i in range(num_tasks)]
    acc_novel_all_ave_avg = [Averager() for i in range(num_tasks)]
    acc_former_all_ave_avg = [Averager() for i in range(num_tasks)]

    acc_avg[0].add(acc)
    if opt.get('details', False):
        acc_former_ave_avg[0].add(acc_former_ave)
        acc_novel_all_ave_avg[0].add(acc_novel_all_ave)
        acc_former_all_ave_avg[0].add(acc_former_all_ave)

    print('*'*60)
    for test_id in range(num_tests):
        for task_id in range(1, num_tasks):
            try:
                opt = deepcopy(opt_old)
            except:
                opt = copy(opt_old)

            opt['name'] = g_opt_name
            opt['test_id'] = test_id
            # Load the model of former session
            # 'task_id = -1' indicates that the program will not load the prototypes, and just load the base model
            opt['task_id'] = task_id - 1

            # The path of model that is updated on former task
            if task_id == 1:
                save_filename_g = f'test{0}_session_{task_id - 1}.pth'
            else:
                save_filename_g = f'test{test_id}_session_{task_id-1}.pth'
            # save_filename_g = f'test{0}_session_{0}.pth'
            save_path_g = osp.join(opt['path']['models'], save_filename_g)
            opt['path']['base_model'] = save_path_g

            #-----------------------------------------------
            model = create_model(opt)

            opt['task_id'] = task_id

            val_classes = random_class_perm[:bases + task_id * num_class_per_task]

            # creating the dataset
            # --------------------------------------------
            for phase, dataset_opt in opt['datasets'].items():
                if phase == 'train':
                    id = opt['train'].get(' ', 0)
                    if num_tests == 1:
                        train_set, train_loader = generate_training_dataset(opt, task_id=task_id, test_id=id)
                    else:
                        train_set, train_loader = generate_training_dataset(opt, task_id=task_id, test_id=test_id)

                if phase == 'val':
                    dataset_opt['all_classes'] = random_class_perm
                    dataset_opt['selected_classes'] = val_classes
                    val_set = create_dataset(dataset_opt=dataset_opt, info=False)
            # --------------------------------------------
            # finetune
            model.incremental_init(train_set, val_set)

            model.incremental_fine_tune(train_dataset=train_set, val_dataset=val_set,
                                             num_epoch=fine_tune_epoch, task_id=task_id, test_id=test_id,
                                             tb_logger=None)

            logger.info('fine-tune procedure is finished!')


            model.incremental_update(novel_dataset=train_set)
            if opt.get('details', False):
                acc, acc_former_ave, acc_former_all_ave, acc_novel_all_ave = model.incremental_test(val_set, task_id, test_id)
                acc_former_ave_avg[task_id].add(acc_former_ave)
                acc_novel_all_ave_avg[task_id].add(acc_novel_all_ave)
                acc_former_all_ave_avg[task_id].add(acc_former_all_ave)
            else:
                acc = model.incremental_test(val_set, task_id, test_id)

            # save the accuracy
            acc_avg[task_id].add(acc)
            model.save(epoch=-1, current_iter=task_id, name=f'test{test_id}_session', dataset=train_set)
            # # reset the opt for creating the model in the next session
            # del opt
            # opt = deepcopy(opt_old)
            # # update the path of saving models
            # # model.set_the_saving_files_path(opt=opt, task_id=task_id, test_id=test_id)
            # # logger.info(f'Successfully saving the model of session {task_id}')
            print('*'*60)

    message = f'--------------------------Final Avg Acc-------------------------'
    logger.info(message)

    for i, acc in enumerate(acc_avg):
        data = acc.obtain_data()
        m = np.mean(data)
        std = np.std(data)
        pm = 1.96 * (std / np.sqrt(len(data)))
        if opt.get('details', False):
            message = f'Session {i+1}: {m*100:.2f}+-{pm*100:.2f}' \
                      f'[acc of former classes: {acc_former_ave_avg[i].item():.4f}]' \
                      f'[acc of former samples in all classes: {acc_former_all_ave_avg[i].item():.4f}]\n' \
                      f'[acc of novel samples in all classes: {acc_novel_all_ave_avg[i].item():.4f}]'
        else:
            message = f'Session {i + 1}: {m * 100:.2f}+-{pm * 100:.2f}'
        logger.info(message)
        if tb_logger:
            tb_logger.add_scalar(f'sessions_acc', acc.item(), i)
            if wandb_logger is not None:
                wandb_logger.log({f'sessions_acc': acc.item()}, step=i)
    logger.info(f'random seed: {seed}')

    print('finish!!')

if __name__ == '__main__':
    main()
