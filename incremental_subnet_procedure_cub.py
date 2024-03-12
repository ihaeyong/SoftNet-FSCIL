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
                   make_exp_dirs, set_random_seed, set_gpu, Averager,
                   safe_load, safe_save)

from dataloader.data_utils import *

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

    parser.add_argument('-dataset', type=str, default='cub200',
                        choices=['mini_imagenet', 'cub200', 'cifar100'])
    parser.add_argument('-dataroot', type=str, default='DataSet/')

    parser.add_argument('--base_class', type=int, default=100)
    parser.add_argument('--num_classes', type=int, default=200)
    parser.add_argument('--way', type=int, default=10)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--sessions', type=int, default=11)
    parser.add_argument('--batch_size_base', type=int, default=256)
    parser.add_argument('--batch_size_new', type=int, default=0)
    parser.add_argument('--test_batch_size', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--autoaug', type=int, default=0)
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--tau_idx', type=int, default=2)
    parser.add_argument('--met_w', type=float, default=1.0)
   
    args = parser.parse_args()
    opt = parse(args.opt, is_train=False, is_incremental=True, args=None)

    # split-FSLL for base session
    args = set_up_datasets(args)
    args.autoaug=0
    args.train=False
    train_set, train_loader, val_set, val_loader = get_base_dataloader(args)

    rank = 0
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

    # define the variables for incremental few-shot learning
    total_classes = opt['datasets']['train']['total_classes']
    bases = opt['train']['bases']
    num_tasks = opt['train']['tasks']
    num_shots = opt['train']['shots']

    # ---------------------
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
    try:
        opt_old = deepcopy(opt)
    except:
        opt_old = copy(opt)

    # Test the session 1 and save the prototypes
    opt['task_id'] = -1
    opt['test_id'] = 0
    model = create_model(opt)
    opt['task_id'] = 0

    val_classes = random_class_perm[:bases]
    selected_classes = random_class_perm[:bases]

    model.incremental_init(train_set, val_set)
    if opt['path'].get('pretrain_prototypes', None) is None:
        model.incremental_update(novel_dataset=train_set)
    if opt.get('Test1', True):
        if opt.get('details', False):
            acc, acc_former_ave, acc_former_all_ave, acc_novel_all_ave = model.incremental_test(val_set, 0, 0)
        else:
            if opt.get('nondist', False):
                train_set.set_aug(False)
                acc = model.validation(train_set, val_loader, 0, tb_logger, mask=None)
            else:
                acc = model.incremental_test(val_set, 0, 0)
    else:
        acc = 0

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
    print("task_id:{}, acc:{}".format(0, acc))

    if wandb_logger is not None:
        task_id = 0
        wandb_logger.log({f'sessions_acc': acc}, step=task_id)
        logger.info(f'sessions{task_id}_acc:{acc}')
        if opt.get('details', False):
            wandb_logger.log({f'sessions_former_acc': acc_former_ave}, step=task_id)
            wandb_logger.log({f'sessions_former_all_acc': acc_former_all_ave}, step=task_id)
            wandb_logger.log({f'sessions_novel_all_acc': acc_novel_all_ave}, step=task_id)
            logger.info(f'sessions{task_id}_former_acc:{acc_former_ave}')
            logger.info(f'sessions{task_id}_former_all_acc:{acc_former_all_ave}')
            logger.info(f'sessions{task_id}_novel_all_acc:{acc_novel_all_ave}')

    print('*'*60)
    per_task_masks = {}
    for test_id in range(num_tests):
        for task_id in range(1, num_tasks):
            # initialize per task_masks
            per_task_masks[task_id] = None

            try:
                opt = deepcopy(opt_old)
            except:
                opt = copy(opt_old)

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

            opt['train']['novel_examplers'] = 0
            #-----------------------------------------------
            model = create_model(opt)
            opt['task_id'] = task_id

            val_classes = random_class_perm[:bases + task_id * num_class_per_task]

            # creating the dataset
            # --------------------------------------------
            args.autoaug=1
            args.train=True
            train_set, train_loader, val_set, val_loader = get_new_dataloader(
                args, task_id)

            # --------------------------------------------
            # finetune
            model.incremental_init(train_set, val_set)

            if opt['subnet_type'] == 'softnet':
                model.incremental_fine_tune(train_loader,
                                            val_dataset=val_set,
                                            num_epoch=fine_tune_epoch,
                                            task_id=task_id,
                                            test_id=test_id,
                                            tb_logger=None,
                                            mask=None)

            logger.info('fine-tune procedure is finished!')

            # get model masks
            per_task_masks[task_id] = model.get_masks(
                sess_id=task_id, # get updated mask
                sparsity=opt['new_sparsity'],
                layer=opt['finetune_layer'])

            del train_set
            del train_loader
            del val_set
            del val_loader

            args.autoaug=0
            args.train=False
            train_set, train_loader, val_set, val_loader = get_new_dataloader(
                args, task_id)

            model.incremental_update(novel_dataset=train_set,mask=per_task_masks[task_id])
            if opt.get('details', False):
                acc, acc_former_ave, acc_former_all_ave, acc_novel_all_ave = model.incremental_test(val_set, task_id, test_id)
                acc_former_ave_avg[task_id].add(acc_former_ave)
                acc_novel_all_ave_avg[task_id].add(acc_novel_all_ave)
                acc_former_all_ave_avg[task_id].add(acc_former_all_ave)
            else:
                if opt.get('nondist', False):
                    #train_set.set_aug(False)
                    acc = model.validation(train_set, val_loader,
                                           task_id, tb_logger,
                                           mask=per_task_masks[task_id])

                else:
                    acc = model.incremental_test(val_set,
                                                 task_id,
                                                 test_id,
                                                 mask=per_task_masks[task_id])

            print('task_id:{}, acc:{}'.format(task_id, acc))
            # save the accuracy
            acc_avg[task_id].add(acc)
            model.save(epoch=-1, current_iter=task_id, name=f'test{test_id}_session',
                       dataset=train_set, mask=per_task_masks, task_id=task_id)
            # # reset the opt for creating the model in the next session


            if wandb_logger is not None:
                wandb_logger.log({f'sessions_acc': acc}, step=task_id)
                logger.info(f'sessions{task_id}_acc:{acc}')
                if opt.get('details', False):
                    wandb_logger.log({f'sessions_former_acc': acc_former_ave}, step=task_id)
                    wandb_logger.log({f'sessions_former_all_acc': acc_former_all_ave}, step=task_id)
                    wandb_logger.log({f'sessions_novel_all_acc': acc_novel_all_ave}, step=task_id)
                    logger.info(f'sessions{task_id}_former_acc:{acc_former_ave}')
                    logger.info(f'sessions{task_id}_former_all_acc:{acc_former_all_ave}')
                    logger.info(f'sessions{task_id}_novel_all_acc:{acc_novel_all_ave}')

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
    print(opt)



if __name__ == '__main__':
    main()
