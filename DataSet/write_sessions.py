import numpy as np
import torch
import random
import os
import os.path as osp
from data import create_dataset
from utils import mkdir_or_exist

# random_seed = 1997
# random_seed = 1993
# random_seed = 1
random_seed = 1997
Random = False

if False:
    total_classes = 100
    bases = 60
    num_tasks = 9
    num_tests = 10
    shots = 5
    dataset_root = './DataSet/FSIL_miniImageNet/miniimagenet'

elif False:
    total_classes = 100
    bases = 60
    num_tasks = 9
    num_tests = 10
    shots = 5
    dataset_root = './DataSet/cifar-100'

elif True:
    total_classes = 200
    bases = 100
    num_tasks = 11
    num_tests = 10
    shots = 5
    dataset_root = './DataSet/CUB_200_2011'

elif False:
    total_classes = 102
    bases = 0
    num_tasks = 6
    num_tests = 10
    shots = 5
    dataset_root = './flowers'

num_class_per_task = int((total_classes - bases) / (num_tasks - 1))

np.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)

if Random:
    random_class_perm = np.random.permutation(total_classes)
else:
    random_class_perm = np.arange(total_classes)

if False:
    dataset_opt = {'name': 'miniImageNet',
                   'type': 'NormalDataset',
                   'total_classes': 100,
                   'dataroot': osp.join(dataset_root, 'Train_all'),
                   'aug': False,
                   'pre_load': False}
elif False:
    dataset_opt = {'name': 'miniImageNet',
                   'type': 'NormalDataset',
                   'total_classes': 100,
                   'dataroot': osp.join(dataset_root, 'train'),
                   'aug': False,
                   'pre_load': False}

elif False:
    dataset_opt = {'name': 'cifar-100',
                   'type': 'NormalDataset',
                   'total_classes': 100,
                   'dataroot': osp.join(dataset_root, 'train'),
                   'aug': False,
                   'pre_load': False}
elif False:
    dataset_opt = {'name': 'cifar-100',
                   'type': 'NormalDataset',
                   'total_classes': 50,
                   'dataroot': osp.join(dataset_root, 'train'),
                   'aug': False,
                   'pre_load': False,
                   'task_id':0}
elif True:
    dataset_opt = {'name': 'cub',
                   'type': 'NormalDataset',
                   'total_classes': 200,
                   'dataroot': osp.join(dataset_root, 'train'),
                   'aug': False,
                   'pre_load': False}

elif False:
    dataset_opt = {'name': 'flowers',
                   'type': 'NormalDataset',
                   'total_classes': 102,
                   'dataroot': osp.join(dataset_root, 'train'),
                   'aug': False,
                   'pre_load': False}

for test_id in range(num_tests):
    for task_id in range(1, num_tasks):
        selected_classes = random_class_perm[bases + (task_id - 1) * num_class_per_task:
                                             bases + task_id * num_class_per_task]

        dataset_opt['all_classes'] = random_class_perm
        dataset_opt['selected_classes'] = selected_classes
        train_set = create_dataset(dataset_opt=dataset_opt, info=False)
        index = train_set.sample_data_index(num_samples_per_class=shots)
        save_root = osp.join(dataset_root, f'Random{Random}_seed{random_seed}_bases{bases}_tasks{num_tasks}_shots{shots}', f'test_{test_id}',
                             f'session_{task_id}')

        save_file = osp.join(save_root, 'index.pt')
        mkdir_or_exist(save_root)

        torch.save(index, save_file)






