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

            val_loader = create_dataloader(
                        val_set,
                        dataset_opt,
                        sampler=None,
                        seed=seed)

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

    layout = {}
    for test_id in range(num_tests):
        layer = {"test_id{}".format(test_id): {}}

        for sess_id in range(9):
            sim_dict = {
                "test_id{}/sess_id{}/similarity".format(test_id, sess_id):
                ["Multiline", ["test_id{}/sess_id{}/similarity/former".format(test_id, sess_id),
                               "test_id{}/sess_id{}/similarity/novel".format(test_id, sess_id),
                               "test_id{}/sess_id{}/similarity/alpha".format(test_id, sess_id),
                               "test_id{}/sess_id{}/similarity/dist_proto".format(test_id, sess_id),
                               "test_id{}/sess_id{}/similarity/dist_former_novel".format(test_id, sess_id)]],}
            pred_dict ={
                "test_id{}/sess_id{}/prediction".format(test_id, sess_id):
                ["Multiline", ["test_id{}/sess_id{}/prediction/label".format(test_id, sess_id),
                               "test_id{}/sess_id{}/prediction/pred".format(test_id, sess_id)]],}
            layer["test_id{}".format(test_id)].update(sim_dict)
            layer["test_id{}".format(test_id)].update(pred_dict)

        acc_dict = {
            "test_id{}/accuracy".format(test_id):
            ["Multiline", ["test_id{}/accuracy/total".format(test_id),
                           "test_id{}/accuracy/former".format(test_id),
                           "test_id{}/accuracy/novel".format(test_id)]],}
        layer["test_id{}".format(test_id)].update(acc_dict)
        layout.update(layer)

    tb_logger.add_custom_scalars(layout)

    if tb_logger:
        log=model.get_current_log()
        for test_id in range(num_tests):
            tb_logger.add_scalar('test_id{}'.format(int(test_id)), acc, 0)

            tb_logger.add_scalar('test_id{}/accuracy/total'.format(test_id), acc, 0)
            tb_logger.add_scalar('test_id{}/accuracy/former'.format(test_id), log['acc_former'], 0)
            tb_logger.add_scalar('test_id{}/accuracy/novel'.format(test_id), log['acc_novel'], 0)

            for i in range(len(log['labels'])):
                tb_logger.add_scalar('test_id{}/sess_id{}/prediction/label'.format(test_id, 0), log['labels'][i], i)
                tb_logger.add_scalar('test_id{}/sess_id{}/prediction/pred'.format(test_id, 0), log['preds'][i], i)

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

                    val_loader = create_dataloader(
                        val_set,
                        dataset_opt,
                        sampler=None,
                        seed=seed)


            # --------------------------------------------
            # finetune
            model.incremental_init(train_set, val_set)

            model.incremental_fine_tune(train_dataset=train_set, val_dataset=val_set,
                                        num_epoch=fine_tune_epoch, task_id=task_id, test_id=test_id,
                                        tb_logger=None, mask=None)

            logger.info('fine-tune procedure is finished!')

            # get model masks
            if False:
                per_task_masks[task_id] = model.get_masks(
                    sess_id=task_id, # get updated mask
                    sparsity=1.0-opt['network_g']['sparsity'],
                    layer=opt['finetune_layer'])
            else:
                per_task_masks[task_id] = None

            model.incremental_update(novel_dataset=train_set,mask=per_task_masks[task_id])
            if opt.get('details', False):
                acc, acc_former_ave, acc_former_all_ave, acc_novel_all_ave = model.incremental_test(val_set, task_id, test_id)
                acc_former_ave_avg[task_id].add(acc_former_ave)
                acc_novel_all_ave_avg[task_id].add(acc_novel_all_ave)
                acc_former_all_ave_avg[task_id].add(acc_former_all_ave)
            else:
                if opt.get('nondist', False):
                    #train_set.set_aug(False)
                    acc = model.validation(train_set, val_loader, task_id, tb_logger, mask=per_task_masks[task_id])

                else:
                    acc = model.incremental_test(val_set, task_id, test_id, mask=per_task_masks[task_id])
                    #model.tsne_plot(test_id, task_id)
                    #model.dist_plot(test_id, task_id)

            print('task_id:{}, acc:{}'.format(task_id, acc))
            # save the accuracy
            acc_avg[task_id].add(acc)
            model.save(epoch=-1, current_iter=task_id, name=f'test{test_id}_session', dataset=train_set, mask=per_task_masks, task_id=task_id)
            # # reset the opt for creating the model in the next session
            # del opt
            # opt = deepcopy(opt_old)
            # # update the path of saving models
            # # model.set_the_saving_files_path(opt=opt, task_id=task_id, test_id=test_id)
            # # logger.info(f'Successfully saving the model of session {task_id}')
            print('*'*60)
            if tb_logger:
                log=model.get_current_log()

                acc_novel_all_ave_avg[task_id].add(log['acc_novel'])
                acc_former_all_ave_avg[task_id].add(log['acc_former'])

                tb_logger.add_scalar('test_id{}'.format(int(test_id)), acc, task_id)

                tb_logger.add_scalar('test_id{}/accuracy/total'.format(test_id), acc, task_id)
                tb_logger.add_scalar('test_id{}/accuracy/former'.format(test_id), log['acc_former'], task_id)
                tb_logger.add_scalar('test_id{}/accuracy/novel'.format(test_id), log['acc_novel'], task_id)

                for i in range(len(log['labels'])):
                    tb_logger.add_scalar('test_id{}/sess_id{}/prediction/label'.format(test_id, task_id), log['labels'][i], i)
                    tb_logger.add_scalar('test_id{}/sess_id{}/prediction/pred'.format(test_id, task_id), log['preds'][i], i)


    message = f'--------------------------Final Avg Acc-------------------------'
    logger.info(message)

    result = {}
    for i,(acc,acc_former,acc_novel) in enumerate(zip(acc_avg, acc_former_all_ave_avg, acc_novel_all_ave_avg)):

        result[str(i)] = {}
        # overall
        data = acc.obtain_data()
        m = np.mean(data)
        std = np.std(data)
        pm = 1.96 * (std / np.sqrt(len(data)))

        result[str(i)]['m'] = m
        result[str(i)]['std'] = std
        result[str(i)]['pm'] = pm

        # former
        data_f = acc_former.obtain_data()
        if len(data_f) > 0:
            m_f = np.mean(data_f)
            std_f = np.std(data_f)
            pm_f = 1.96 * (std_f / np.sqrt(len(data_f)))
        else:
            m_f = 0
            std_f = 0
            pm_f = 0

        result[str(i)]['m_f'] = m_f
        result[str(i)]['std_f'] = std_f
        result[str(i)]['pm_f'] = pm_f

        # novel
        data_n = acc_novel.obtain_data()
        if len(data_n) > 0:
            m_n = np.mean(data_n)
            std_n = np.std(data_n)
            pm_n = 1.96 * (std_n / np.sqrt(len(data_n)))
        else:
            m_n = 0
            std_n = 0
            pm_n = 0

        result[str(i)]['m_n'] = m_n
        result[str(i)]['std_n'] = std_n
        result[str(i)]['pm_n'] = pm_n

        if opt.get('details', False):
            message = f'Session {i+1}: {m*100:.2f}+-{pm*100:.2f}' \
                      f'[acc of former classes: {acc_former_ave_avg[i].item():.4f}]' \
                      f'[acc of former samples in all classes: {acc_former_all_ave_avg[i].item():.4f}]\n' \
                      f'[acc of novel samples in all classes: {acc_novel_all_ave_avg[i].item():.4f}]'
        else:
            message = f'Session {i + 1}: {m * 100:.2f}+-{pm * 100:.2f}, {m_f * 100:.2f}+-{pm_f * 100:.2f}, {m_n * 100:.2f}+-{pm_n * 100:.2f}'
        logger.info(message)
        if tb_logger:
            tb_logger.add_scalar(f'sessions_acc', acc.item(), i)
            tb_logger.add_scalar(f'sessions_former_acc', acc_former.item(), i)
            tb_logger.add_scalar(f'sessions_novel_acc', acc_novel.item(), i)
            if wandb_logger is not None:
                wandb_logger.log({f'sessions_acc': acc.item()}, step=i)
    logger.info(f'random seed: {seed}')

    log_file = './tb_logger_{}/'.format(opt['datasets']['train']['name']) + opt['name'] + '/result.npy'
    safe_save(log_file, result)

    print('finish!!')
    print(opt)

if __name__ == '__main__':
    main()
