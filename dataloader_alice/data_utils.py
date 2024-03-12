import numpy as np
import torch
from torchvision import transforms
from .sampler import CategoriesSampler
from .loader import TwoCropsTransform
from PIL import Image


def set_up_datasets(args):
    if args.dataset == 'cifar100':
        from .cifar100 import cifar as Dataset
        args.base_class = 60
        args.num_classes = 100
        args.way = 5
        args.shot = 5
        args.sessions = 9
        args.img_dim = 32
    if args.dataset == 'cub200':
        from .cub200 import cub200 as Dataset
        args.base_class = 100
        args.num_classes = 200
        args.way = 10
        args.shot = 5
        args.sessions = 11
        args.img_dim = 84
    if args.dataset == 'mini_imagenet':
        from .miniimagenet import miniimagenet as Dataset
        args.base_class = 60
        args.num_classes = 100
        args.way = 5
        args.shot = 5
        args.sessions = 9
        args.img_dim = 224

    args.Dataset = Dataset
    return args


# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
def get_train_dataloader(args, session):
    if session == 0:
        trainset, trainloader = get_base_train_dataloader(args)
    else:
        trainset, trainloader = get_new_dataloader(args, session)
    return trainset, trainloader


def get_base_train_dataloader(args):
    class_index = np.arange(args.base_class)
    # class_index = np.arange(args.num_classes) # load all the class data

    if args.dataset == 'cifar100':
        trainset = args.Dataset.CIFAR100(root=args.data_root, train=True, download=True, index=class_index, base_sess=True, two_images=args.data_transform)
    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.data_root, train=True, index=class_index, base_sess=True, two_images=args.data_transform)
    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.data_root, train=True, index=class_index, base_sess=True, two_images=args.data_transform)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    return trainset, trainloader


def get_new_dataloader(args, session):
    txt_path_list = []
    txt_path = "./DataSet/DATA_TOPIC/" + args.dataset + "/session_" + str(session + 1) + '.txt'
    txt_path_list.append(txt_path)

    if args.dataset == 'cifar100':
        class_index = open(txt_path).read().splitlines()
        trainset = args.Dataset.CIFAR100(root=args.data_root, train=True, download=False, index=class_index, base_sess=False, two_images=args.data_transform)
    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.data_root, train=True, index_path=txt_path_list, base_sess=False, two_images=args.data_transform)
    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.data_root, train=True, index_path=txt_path_list, base_sess=False, two_images=args.data_transform)

    if args.batch_size_new == 0:
        batch_size_new = trainset.__len__()
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    else:
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_new, shuffle=True, num_workers=args.num_workers, drop_last=True)
    return trainset, trainloader


# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
def get_incremental_dataset_fs(args, session=None):
    class_index = []
    if session == None:
        session = args.sessions

    print('session: {0}'.format(session))
    txt_path_list = []
    if False:
        for i in range(session + 1):
            if i == 0:
                txt_path = "./DataSet/DATA_TOPIC/" + args.dataset + '/session_{0}'.format(str(i + 1)) + '.txt'
            else:
                txt_path = "./DataSet/DATA_TOPIC/" + args.dataset + '/session_{0}'.format(str(i + 1)) + '.txt'
            temp_class_index = open(txt_path).read().splitlines()
            for j in range(len(temp_class_index)):
                class_index.append(temp_class_index[j])
            txt_path_list.append(txt_path)
    else:
        # revised by haeyong.k
        txt_path = "./DataSet/DATA_TOPIC/" + args.dataset + '/session_{0}'.format(str(session + 1)) + '.txt'
        temp_class_index = open(txt_path).read().splitlines()
        for j in range(len(temp_class_index)):
            class_index.append(temp_class_index[j])
        txt_path_list.append(txt_path)

    print('number of images: {0}'.format(len(class_index)))

    print('~~~~~~~~ training dataset ~~~~~~~~')
    if args.dataset == 'cifar100':
        trainset = args.Dataset.CIFAR100(root=args.data_root, train=True, download=True, index=class_index, base_sess=False, two_images=False, validation=False)
    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.data_root, train=True, index_path=txt_path_list, base_sess=False, two_images=False, validation=False)
    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.data_root, train=True, index_path=txt_path_list, base_sess=False, two_images=False, validation=False)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    else:
        train_sampler = None

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, sampler=train_sampler, pin_memory=True)

    print('~~~~~~~~ testing dataset ~~~~~~~~')
    class_new = get_session_classes(args, session)  # test on all encountered classes

    if args.dataset == 'cifar100':
        testset = args.Dataset.CIFAR100(root=args.data_root, train=False, download=False, index=class_new, base_sess=False, two_images=False, validation=False)
    if args.dataset == 'cub200':
        testset = args.Dataset.CUB200(root=args.data_root, train=False, index=class_new, base_sess=False, two_images=False, validation=False)
    if args.dataset == 'mini_imagenet':
        testset = args.Dataset.MiniImageNet(root=args.data_root, train=False, index=class_new, base_sess=False, two_images=False, validation=False)

    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=100, shuffle=False, num_workers=args.workers, pin_memory=True)
    return trainset, trainloader, testset, testloader


# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
def get_validation_dataloader(args):
    # class_index = np.arange(args.num_classes)
    class_index = np.arange(args.base_class)

    if args.dataset == 'cifar100':
        trainset = args.Dataset.CIFAR100(root=args.data_root, train=True, download=True, index=class_index, base_sess=True, two_images=False, validation=True)
        testset = args.Dataset.CIFAR100(root=args.data_root, train=False, download=False, index=class_index, base_sess=False, two_images=False, validation=True)
    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.data_root, train=True, index=class_index, base_sess=True, two_images=False, validation=True)
        testset = args.Dataset.CUB200(root=args.data_root, train=False, index=class_index, base_sess=False, two_images=False, validation=True)
    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.data_root, train=True, index=class_index, base_sess=True, two_images=False, validation=True)
        testset = args.Dataset.MiniImageNet(root=args.data_root, train=False, index=class_index, base_sess=False, two_images=False, validation=True)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    return trainset, trainloader, testset, testloader


# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
def get_session_classes(args=None, session=None, opt=None):

    if args is not None:
        class_list=np.arange(args.base_class + session * args.way)
    else:
        class_list=np.arange(opt['train']['bases'] + session * opt['train']['way'])

    return class_list
