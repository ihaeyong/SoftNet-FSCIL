import yaml
from collections import OrderedDict
from os import path as osp

def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def parse(opt_path, is_train=True, is_incremental=False, args=None):
    """Parse option file.

    Args:
        opt_path (str): Option file path.
        is_train (boolean): Indicate whether in normal training stage. Default: True.
        is_incremental (boolean): Indicate whether in class incremental procedure

    Returns:
        (dict): Options.

    """
    if is_train and is_incremental:
        raise ValueError('is_train and is_incremental can not both be true')

    with open(opt_path, mode='r') as f:
        Loader, _ = ordered_yaml()
        opt = yaml.load(f, Loader=Loader)

    if args is not None:
        # -------- cub hyper params learning -------
        tau_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]

        opt['train']['fine_tune_epoch'] = args.epoch
        opt['train']['cpn_opt']['tau'] = tau_list[int(args.tau_idx)]
        opt['train']['metric_opt']['w'] = args.met_w
        opt['name'] = opt['name'] + '_epoch_' + str(args.epoch) + '_tau_' + str(tau_list[int(args.tau_idx)]) + '_met_w_' + str(args.met_w)
        # ------------------------------------------

    opt['is_train'] = is_train
    opt['is_incremental'] = is_incremental

    # datasets
    for phase, dataset in opt['datasets'].items():
        # for several datasets, e.g., test_1, test_2
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        if 'scale' in opt:
            dataset['scale'] = opt['scale']
        if dataset.get('dataroot') is not None:
            dataset['dataroot'] = osp.expanduser(dataset['dataroot'])
        dataset['transformer_agu'] = opt['transformer_agu']
        dataset['transformer'] = opt['transformer']

        if 'transformer_color_agu' in opt.keys():
            dataset['transformer_color_agu'] = opt['transformer_color_agu']

    # paths
    for key, path in opt['path'].items():
        if path and key != 'strict_load':
            opt['path'][key] = osp.expanduser(path)
    opt['path']['root'] = osp.abspath(
        osp.join(__file__, osp.pardir, osp.pardir))
    if is_train: # normal training stage
        dataset_name = opt['datasets']['train']['name']
        bases = opt['train']['bases']
        exp_str = f'{dataset_name}_bases{bases}'
        if 'noBuffer' in opt_path:
            exp_str = exp_str + '_noBuffer'
        experiments_root = osp.join(opt['path']['root'], 'exp', exp_str,
                                    opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'models')
        opt['path']['training_states'] = osp.join(experiments_root,
                                                  'training_states')
        opt['path']['log'] = experiments_root
        opt['path']['visualization'] = osp.join(experiments_root,
                                                'visualization')

        # change some options for debug mode
        if 'debug' in opt['name']:
            opt['val']['val_freq'] = opt['val'].get('debug_val_freq', 10)
            opt['logger']['print_freq'] = 1

    if is_incremental:  # incremental training
        dataset_name = opt['datasets']['train']['name']
        bases = opt['train']['bases']
        exp_str = f'{dataset_name}_bases{bases}'
        incremental_root = osp.join(opt['path']['root'], 'increment', exp_str, opt['name'])
        opt['path']['incremental_root'] = incremental_root
        opt['path']['training_states'] = osp.join(incremental_root,
                                                  'training_states')
        opt['path']['log'] = incremental_root
        opt['path']['models'] = osp.join(incremental_root, 'models')
        opt['path']['prototypes'] = osp.join(incremental_root, 'prototypes')
        opt['path']['feat_buffer'] = osp.join(incremental_root, 'feat_buffer')

    if not is_incremental and not is_train: # normal test stage
        results_root = osp.join(opt['path']['root'], 'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root

    return opt


def dict2str(opt, indent_level=1):
    """dict to string for printing options.

    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.

    Return:
        (str): Option string for printing.
    """
    msg = '\n'
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_level * 2) + k + ':['
            msg += dict2str(v, indent_level + 1)
            msg += ' ' * (indent_level * 2) + ']\n'
        else:
            msg += ' ' * (indent_level * 2) + k + ': ' + str(v) + '\n'
    return msg

