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

from dataloader_alice.data_utils import * # topic split
from data import create_dataloader, create_dataset, create_sampler
from methods import create_model
from utils.options import dict2str, parse
from utils import (MessageLogger, get_env_info, get_root_logger,
                   init_tb_logger, init_wandb_logger, check_resume,
                   make_exp_dirs, set_random_seed, set_gpu, Averager,
                   safe_load, safe_save)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt',type=str, required=True, help='Path to option YAML file.')

    # for alice, topic
    parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
    parser.add_argument('--batch_size_new', type=int, default=0, help='set 0 will use all the availiable training image for new')
    parser.add_argument('--workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--data_transform', action='store_true', help='Whether use 2 set of transformed data per input image to do calculation.')
    parser.add_argument('--data_root', type=str, help='path to dataset directory')
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['mini_imagenet', 'cub200', 'cifar100'])

    parser.add_argument('--current_session', default=0, type=int)
    parser.add_argument('--used_img', default=500, type=int)  # 500, 5, 1
    parser.add_argument('--balanced', default=0, type=int)

    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                        'N processes per node, which has N GPUs. This is the '
                        'fastest way to use PyTorch for either single node or'
                        'multi node data parallel training')

    args = parser.parse_args()
    opt = parse(args.opt, is_train=False, is_incremental=True)
    assert opt['datasets']['train']['name'] in ['cifar100', 'cub200', 'mini_imagenet']
    args.dataset = opt['datasets']['train']['name']
    args.data_root = opt['datasets']['train']['dataroot']
    args.used_img = opt['train']['shots']
    args = set_up_datasets(args)

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    print(vars(args))

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

    for phase, dataset_opt in opt['datasets'].items():
        # load training data for topic split
        dataset_opt['task_id'] = 0
    args.current_session = dataset_opt['task_id']
    train_set, train_loader, val_set, val_loader = get_incremental_dataset_fs(args, session=args.current_session)
    print('length of the trainset: {0}'.format(len(train_set)))

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

            print(opt['name'])
            opt['test_id'] = test_id
            # Load the model of former session
            # 'task_id = -1' indicates that the program will not load the prototypes, and just load the base model
            opt['task_id'] = task_id - 1

            # The path of model that is updated on former task
            if task_id == 1:
                save_filename_g = f'test{0}_session_{task_id - 1}.pth'
                save_filename_e = f'test{0}_session_extra_{task_id - 1}.pth'
            else:
                save_filename_g = f'test{test_id}_session_{task_id-1}.pth'
                save_filename_e = f'test{test_id}_session_extra_{task_id-1}.pth'

            # save_filename_g = f'test{0}_session_{0}.pth'
            save_path_g = osp.join(opt['path']['models'], save_filename_g)
            opt['path']['base_model'] = save_path_g

            save_path_e = osp.join(opt['path']['models'], save_filename_e)
            opt['path']['extra_model'] = save_path_e
            #-----------------------------------------------
            model = create_model(opt)
            opt['task_id'] = task_id

            val_classes = random_class_perm[:bases + task_id * num_class_per_task]

            # creating the dataset
            # --------------------------------------------
            # topic, alice split
            args.current_session = task_id
            train_set, train_loader, val_set, val_loader = get_incremental_dataset_fs(args, session=args.current_session)
            print('length of the trainset: {0}'.format(len(train_set)))

            # --------------------------------------------
            # finetune
            model.incremental_init(train_set, val_set)

            if opt['task_id'] > 0:
                model.incremental_fine_tune(train_dataset=train_set,
                                            train_loader=train_loader,
                                            val_dataset=val_set,
                                            val_loader=val_loader,
                                            num_epoch=fine_tune_epoch,
                                            task_id=task_id,
                                            test_id=test_id,
                                            tb_logger=None, mask=None)

            logger.info('fine-tune procedure is finished!')

            # get model masks
            per_task_masks[task_id] = None

            model.incremental_update(novel_dataset=train_set,mask=per_task_masks[task_id])
            if opt.get('details', False):
                acc, acc_former_ave, acc_former_all_ave, acc_novel_all_ave = model.incremental_test(val_set, task_id, test_id)
                acc_former_ave_avg[task_id].add(acc_former_ave)
                acc_novel_all_ave_avg[task_id].add(acc_novel_all_ave)
                acc_former_all_ave_avg[task_id].add(acc_former_all_ave)
            else:
                if opt.get('nondist', False):
                    train_set.set_aug(flag=False)
                    acc = model.validation(train_set, val_loader, task_id, tb_logger, mask=per_task_masks[task_id])

                else:
                    acc = model.incremental_test(val_set, task_id, test_id, mask=per_task_masks[task_id])

            print('task_id:{}, acc:{}'.format(task_id, acc))
            # save the accuracy
            acc_avg[task_id].add(acc)
            model.save(epoch=-1, current_iter=task_id, name=f'test{test_id}_session', dataset=train_set, mask=per_task_masks, task_id=task_id)

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
