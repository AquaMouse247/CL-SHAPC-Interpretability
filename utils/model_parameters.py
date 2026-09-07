# iTAML Imports
from models.iTAML.basic_net import *

def get_algorithm_args(algorithm, dataset):
    args = None
    if algorithm == "iTAML":
        args = iTAMLArgs
    if algorithm == "RPSnet":
        args = MnistArgs
    if algorithm == "foster":
        args = fosterArgs
    if algorithm == "memo":
        args = memoArgs
    if algorithm == "der":
        args = derArgs
    if algorithm == "icarl":
        args = icarlArgs
    if algorithm == "dsal":
        args = dsalArgs
    if algorithm == "tagfex":
        args = tagfexArgs
    if algorithm == "xder":
        args = xderArgs

    if algorithm in pycil_algs:
        if dataset == "mnist":
            if algorithm == "memo":
                args['convnet_type'] = "memo_resnet32mnist"
            else:
                args['convnet_type'] = "resnet32mnist"
        if dataset == "cifar100" and algorithm == "ds-al":
            args['convnet_type'] = "resnet18"
        elif dataset == "imagenet200":
            if algorithm == "memo":
                args['convnet_type'] = "memo_resnet18"
            else:
                args['convnet_type'] = "resnet18"
            args['dataset'] = "imagenet200"

    return args

# RPSnet Imports
class iTAMLArgs:
    checkpoint = "results/cifar100/meta_mnist_T5_47"
    savepoint = "models/" + "/".join(checkpoint.split("/")[1:])
    data_path = "../Datasets/MNIST/"
    num_class = 10
    class_per_task = 2
    num_task = 5
    test_samples_per_class = 1000
    dataset = "mnist"
    optimizer = 'sgd'

    epochs = 20
    lr = 0.1
    train_batch = 256
    test_batch = 256
    workers = 16
    sess = 0
    schedule = [5, 10, 15]
    gamma = 0.5
    random_classes = False
    validation = 0
    memory = 2000
    mu = 1
    beta = 0.5
    r = 1

# RPSnet Imports
from models.RPSnet.rps_net import RPS_net_mlp, RPS_net_cifar, RPS_net, generate_path


class MnistArgs:
    epochs = 10
    checkpoint = "results/mnist/RPS_net_mnist"
    savepoint = "results/mnist/pathnet_mnist"
    dataset = "MNIST"
    num_class = 10
    class_per_task = 2
    M = 8
    L = 9
    N = 1
    lr = 0.001
    train_batch = 128
    test_batch = 128
    workers = 16
    resume = False
    arch = "res-18"
    start_epoch = 0
    evaluate = False
    sess = 0
    test_case = 0
    schedule = [6, 8, 16]
    gamma = 0.5
    rigidness_coff = 2.5
    jump = 1


class Cifar10Args:
    epochs = 10
    #    epochs = 2
    checkpoint = "results/cifar10/RPS_net_cifar10"
    savepoint = "results/cifar10/pathnet_cifar10"
    dataset = "CIFAR10"
    num_class = 10
    class_per_task = 2
    M = 8
    L = 9
    N = 1
    lr = 0.001
    train_batch = 128
    test_batch = 128
    workers = 16
    resume = False
    arch = "res-18"
    start_epoch = 0
    evaluate = False
    sess = 0
    test_case = 0
    schedule = [6, 8, 16]
    gamma = 0.5
    rigidness_coff = 2.5
    jump = 1


class Cifar100Args:
    checkpoint = "results/cifar100/RPS_CIFAR_M8_J1"
    labels_data = "prepare/cifar100_10.pkl"
    savepoint = ""

    num_class = 100
    class_per_task = 10
    M = 8
    jump = 2
    rigidness_coff = 2.5
    dataset = "CIFAR"

    epochs = 100
    epochs = 1
    L = 9
    N = 1
    lr = 0.001
    train_batch = 128
    test_batch = 128
    workers = 16
    resume = False
    arch = "res-18"
    start_epoch = 0
    evaluate = False
    sess = 0
    test_case = 0
    schedule = [20, 40, 60, 80]
    gamma = 0.5


class SalGenArgs:
    algorithm = "RPSnet"
    dataset = "mnist"
    args = None
    desired_classes = [0, 1]
    class_per_task = 2
    num_class = 10
    distill = False

# XDER Imports
import os
original_cwd = os.getcwd()
from models.xder.models.xder import XDer
from models.xder.backbone import get_backbone_class
from models.xder.datasets import get_dataset_class
from models.xder.utils.checkpoints import mammoth_load_checkpoint
os.chdir(original_cwd)
xderArgs = {'loadcheck': None, 'dataset': 'seq-cifar10', 'model': 'xder', 'backbone': 'resnet18',
            'load_best_args': False, 'dataset_config': None, 'model_config': 'default',
            'buffer_size': 2000, 'minibatch_size': 128, 'alpha': 0.6, 'beta': 0.9,
            'simclr_temp': 5.0, 'gamma': 0.85, 'simclr_batch_size': 64, 'simclr_num_aug': 4,
            'lambd': 0.05, 'constr_eta': 0.1, 'constr_margin': 0.3, 'dp_weight': 0.0,
            'past_constraint': False, 'future_constraint': True, 'align_bn': True,
            'transform_type': 'weak', 'num_classes': 10, 'num_filters': 64, 'seed': 1993,
            'permute_classes': False, 'base_path': './data/', 'checkpoint_path': './checkpoints/',
            'results_path': 'results/', 'device': 'cuda:0', 'notes': None, 'eval_epochs': None,
            'non_verbose': False, 'disable_log': False, 'num_workers': None, 'enable_other_metrics': False,
            'debug_mode': False, 'inference_only': False, 'code_optimization': 0, 'distributed': 'no',
            'savecheck': 'task', 'save_checkpoint_mode': 'safe',
            'ckpt_name': 'xder_seq-cifar10_None_2000_2_20251201-035705_93385ae6', 'start_from': None,
            'stop_after': None, 'save_after_interrupt': True, 'wandb_name': None, 'wandb_entity': None,
            'wandb_project': None, 'lr': 0.03, 'batch_size': 128, 'label_perc': 1.0,
            'label_perc_by_class': 1.0, 'joint': 0, 'eval_future': False, 'custom_task_order': None,
            'custom_class_order': None, 'validation': None, 'validation_mode': 'current',
            'fitting_mode': 'epochs', 'early_stopping_patience': 5, 'early_stopping_metric': 'loss',
            'early_stopping_freq': 1, 'early_stopping_epsilon': 1e-06, 'n_epochs': 50, 'n_iters': None,
            'optimizer': 'sgd', 'optim_wd': 0.0, 'optim_mom': 0.0, 'optim_nesterov': False,
            'drop_last': False, 'lr_scheduler': None, 'scheduler_mode': 'epoch', 'lr_milestones': [],
            'sched_multistep_lr_gamma': 0.1, 'noise_type': 'symmetric', 'noise_rate': 0.0,
            'disable_noisy_labels_cache': False, 'cache_path_noisy_labels': 'noisy_labels',
            'conf_jobnum': '93385ae6-a3dc-4e80-9855-1a55c58e472a',
            'conf_timestamp': '2025-12-01 03:57:05.038568', 'conf_host': 'b752ea1fefda',
            'conf_git_hash': 'ea967c34ac56d97d52e4baeb82274169cdface6e', 'nowand': 1}


# PyCIL Imports
pycil_algs = ["der", "foster", "memo", "icarl", "ds-al", "tagfex"]
from models.PyCIL.models import foster
from models.PyCIL.utils import factory
from models.PyCIL.utils.inc_net import FOSTERNet, AdaptiveNet, DERNet, IncrementalNet, SimpleCosineIncrementalNet
from models.PyCIL.utils.inc_net import DSALNet, ACILNet, BaseNet, TagFexNet

fosterArgs = {'config': './exps/fostertest.json', 'prefix': 'cil', 'dataset': 'cifar100',
              'memory_size': 2000, 'memory_per_class': 20, 'fixed_memory': True,
              'shuffle': False, 'init_cls': 10, 'increment': 10, 'model_name': 'foster',
              'convnet_type': 'resnet32', 'device': ['0'], 'seed': [1993], 'beta1': 0.96,
              'beta2': 0.97, 'oofc': 'ft', 'is_teacher_wa': False, 'is_student_wa': False,
              'lambda_okd': 1, 'wa_value': 1, 'init_epochs': 1, 'init_lr': 0.1,
              'init_weight_decay': 0.0005, 'boosting_epochs': 1, 'compression_epochs': 1,
              'lr': 0.1, 'batch_size': 128, 'weight_decay': 0.0005, 'num_workers': 8, 'T': 2}

memoArgs = {"prefix": "benchmark", "dataset": "cifar100", "memory_size": 2000,
            "memory_per_class":20, "fixed_memory": False, "shuffle": False, "init_cls": 10,
            "increment": 10, "model_name": "memo", "convnet_type": "memo_resnet32",
            "train_base": True, "train_adaptive": False, "debug": False, "skip": False,
            "device": ["0"], "seed":[1993], "scheduler": "steplr", "init_epoch": 200,
            "t_max": None, "init_lr" : 0.1, "init_weight_decay" : 5e-4, "init_lr_decay" : 0.1,
            "init_milestones" : [60,120,170], "milestones" : [80,120,150], "epochs": 170,
            "lrate" : 0.1, "batch_size" : 128, "weight_decay" : 2e-4, "lrate_decay" : 0.1,
            "alpha_aux" : 1.0}

derArgs = {"prefix": "reproduce", "dataset": "cifar10", "memory_size": 2000,
           "memory_per_class": 20, "fixed_memory": False, "shuffle": False, "init_cls": 2,
           "increment": 2, "model_name": "der", "convnet_type": "resnet32",
           "device": ["0"], "seed": [1993]}

icarlArgs = {"prefix": "reproduce", "dataset": "cifar10", "memory_size": 2000,
             "memory_per_class": 20, "fixed_memory": False, "shuffle": False, "init_cls": 2,
             "increment": 2, "model_name": "icarl", "convnet_type": "resnet32",
             "device": ["0"], "seed": [1993]}

dsalArgs = {"model_name": "ds-al", "prefix": "DS-AL", "memory_size": 0, "dataset": "cifar10",
            "seed": [1993], "shuffle": False, "device": ["0"], "convnet_type": "resnet32",
            "init_cls": 2, "increment": 2, "num_workers": 16, "init_batch_size": 128,
            "IL_batch_size": 4096, "inplace_repeat": 1,

            "configurations": {
                "cifar10": {
                    "buffer_size": 4096,
                    "gamma": 0.1,
                    "gamma_comp": 0.1,
                    "compensation_ratio": 0.6,
                    "init_weight_decay": 5e-4,
                    "scheduler": {
                        "type": "MultiStep",
                        "init_lr": 0.1,
                        "init_epochs": 160,
                        "warmup": 0,
                        "milestones": [120, 140],
                        "decay": 0.1
                    }
                },
                "cifar100": {
                    "buffer_size": 8192,
                    "gamma": 0.1,
                    "gamma_comp": 0.1,
                    "compensation_ratio": 0.6,
                    "init_weight_decay": 5e-4,
                    "scheduler": {
                        "type": "MultiStep",
                        "init_lr": 0.1,
                        "init_epochs": 160,
                        "warmup": 0,
                        "milestones": [120, 140],
                        "decay": 0.1
                    }
                }
            }
        }

tagfexArgs = {"prefix": "reproduce", "dataset": "cifar100", "memory_size": 2000,
              "memory_per_class": 20, "fixed_memory": False, "shuffle": False,
              "init_cls": 10, "increment": 10, "model_name": "tagfex",
              "convnet_type": "resnet32", "device": ["0"], "seed": [1993],
              "init_interpolation_factor": 0.95, "infonce_temp":0.2, "kd_temp":2,
              "infonce_kd_temp":0.2, "ta_convnet_type":"resnet18", "proj_hidden_dim":2048,
              "proj_output_dim":1024, "attn_num_heads":8, "contrast_factor":1,
              "trans_cls_factor":1, "transfer_factor":1, "aux_factor":2,
              "contrast_kd_factor":2, "aug":2}
