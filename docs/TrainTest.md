# Training and Testing

[English](TrainTest.md)

Please run the commands in the root path of `FS-IL`. <br>
In general, both the training and testing include the following steps:

1. Prepare datasets. 
1. Modify config files. The config files are under the `options` folder. For more specific configuration information, please refer to [Config](Config.md)
1. Run commands. Use [Training Commands](#Training-Commands) or [Testing Commands](#Testing-Commands) accordingly.

#### Catalog

1. [Training Commands (Training the base model)](#Training-Commands)

1. [Incremental Testing Commands](#Testing-Commands)


## Training Commands

### Baseline
> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0 \\\
> python train.py -opt options/train/cifar/FSIL/FSIL_train_res18_cifar_CFRPModel_baseline_SGD.yml

### ICaRL
> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0 \\\
> python train.py -opt options/train/cifar/FSIL/FSIL_train_res18_cifar_ICaRL_SGD_01.yml

### Rebalance
> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0 \\\
> python train.py -opt options/train/cifar/FSIL/FSIL_train_res18_cifar_NCM_SGD.yml

### FSLL
> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0 \\\
> python train.py -opt options/train/cifar/FSIL/FSIL_train_res18_cifar_CFRPModel_baseline_SGD.yml

### cRT (including testing)
> python run_cifar_ub.py

### F2M
> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0 \\\
> python train.py -opt options/train/cifar/FSIL/FSIL_train_res18_cifar_CFRPModel_AG_ProtoFix_SGD_01.yml

## Incremental Testing Commands

### Baseline
> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0 \\\
> python incremental_procedure.py -opt options/train/cifar/FSIL/incremental_res18_cifar_baseline_5shots_SGD.yml

### ICaRL
> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0 \\\
> python incremental_procedure_buffer.py -opt options/train/cifar/FSIL/incremental_res18_cifar_iCaRL_5shots_SGD.yml

### Rebalance
> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0 \\\
> python incremental_procedure_buffer.py -opt options/train/cifar/FSIL/incremental_res18_cifar_NCM_5shots_SGD.yml

### FSLL
> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0 \\\
> python incremental_procedure.py -opt options/train/cifar/FSIL/incremental_res18_cifar_FSLLModel_baseline_SGD.yml

### F2M
> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0 \\\
> python train.py -opt options/train/cifar/FSIL/FSIL_train_res18_cifar_CFRPModel_AG_ProtoFix_SGD_01.yml