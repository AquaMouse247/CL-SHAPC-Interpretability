### Load compare dict
import numpy as np
import scipy.io
import os
import torch
from torch.utils.data import DataLoader

# Custom Imports
from utils.setup_args import SHAPArgs, create_shap_value_filepath, create_preds_savepath, get_subset_filepath
from utils.load_models import load_model, load_meta_models, generate_predictions
from utils.model_parameters import pycil_algs

import utils.shap_dataloader as sdl
from models.RPSnet.rps_net import generate_path



algorithm = "RPSnet"
dataset = "cifar10"
shapArgs = SHAPArgs(algorithm, dataset)
device = "cuda:0" if torch.cuda.is_available() else "cpu"


first_last_only = True
subset_testing = True
subset_num = 21
filepath = create_shap_value_filepath(shapArgs, first_last_only, subset_testing, subset_num) + ".npy"
preds_savepath = create_preds_savepath(shapArgs)

num_tasks = shapArgs.dataset_params.num_task
num_class = shapArgs.dataset_params.num_class
cls_per_task = shapArgs.dataset_params.class_per_task
shap_samples = shapArgs.dataset_params.shap_samples

if algorithm == "iTAML":
    loaded_models = []
    for i in range(num_tasks):
        loaded_models.extend(load_meta_models(dataset, i))
else:
    loaded_models = [load_model(algorithm, dataset, i, shapArgs=shapArgs).to(device) for i in range(num_tasks)]

shap_values_loaded = np.load(filepath, allow_pickle=True)  # ['shap_dict']
num_imgs = len(shap_values_loaded[()].keys())
shap_dict = {}
for i in range(num_imgs):
    shap_dict[f'{i}'] = shap_values_loaded[()][f'{i}']

# Get test dataset
if subset_testing:
    shap_subset = torch.load(get_subset_filepath(shapArgs), weights_only=False)[subset_num-1]
    shap_dataloader = DataLoader(shap_subset, batch_size=10000, shuffle=False)
    *_, test_imgs, test_labels = next(iter(shap_dataloader))
    print("Loaded Imgs:", len(test_imgs))
else:
    sal_dataloader = sdl.ShapDataloader(shapArgs)
    # Get test dataset
    for i in range(num_tasks):
        if dataset == "cifar100":
            sal_imgs, sal_labels, _, STD, MEAN = sal_dataloader.load_data(range(i * 10, (i * 10) + 10), 20, batch_size=10000)
        elif dataset == "imagenet200":
            sal_imgs, sal_labels, _, STD, MEAN = sal_dataloader.load_data(range(i * 20, (i * 20) + 20), 20, batch_size=10000)
        else:
            ###---Updated to take initial classes learned into account---###
            if i == 0:
                sal_imgs, sal_labels, _, STD, MEAN = sal_dataloader.load_data(range(shapArgs.dataset_params.init_cls),
                                                                              shapArgs.dataset_params.shap_samples, batch_size=10000)
            else:
                sal_imgs, sal_labels, _, STD, MEAN = sal_dataloader.load_data(range((shapArgs.dataset_params.init_cls-cls_per_task)+i*cls_per_task,
                                                                                shapArgs.dataset_params.init_cls+(i*cls_per_task)),
                                                                                shapArgs.dataset_params.shap_samples, batch_size=10000)
            ###----------------------------------------------------------###
        print("Len of sal_imgs:", len(sal_imgs))
        if i == 0:
            test_imgs, test_labels = sal_imgs, sal_labels
        else:
            test_imgs = torch.cat((test_imgs, sal_imgs), 0)
            test_labels = torch.cat((test_labels, sal_labels), 0)

#print("Len of sal_imgs:", len(test_imgs))
test_imgs, test_labels = test_imgs.to(device), test_labels.to(device)

'''
if algorithm == "iTAML":
    loaded_models = []
    for i in range(num_tasks):
        loaded_models.extend(load_meta_models(dataset, i))
else:
    loaded_models = [load_model(algorithm, dataset, i, shapArgs=shapArgs).to(device) for i in range(num_tasks)]
'''

samples = range(shap_samples*(num_class-cls_per_task))

for sample in samples:
    test_sample = shap_dict[f'{sample}']
    test_sess = list(test_sample.keys())
    test_sess.remove(test_sess[1])
    print(test_sess)
    ses = int(test_sess[0][-1])

    #sample_multiplier = cls_per_task * shap_samples

    # Get test image
    test_img = test_imgs[sample].unsqueeze(0)
    test_label = test_labels[sample]
    models = [loaded_models[int(test_sess[0][-1])], loaded_models[int(test_sess[-1][-1])]]

    # Generate predictions
    if algorithm == "RPSnet":
        infer_path = generate_path(ses, dataset, shapArgs.algorithm_args)
        preds = [generate_predictions(algorithm, models[0], int(test_sess[0][-1]), test_img, infer_path=infer_path, cls_per_task=cls_per_task),
                 generate_predictions(algorithm, models[1], int(test_sess[-1][-1]), test_img, infer_path=infer_path, cls_per_task=cls_per_task)]
    else:
        preds = [generate_predictions(algorithm, models[0], int(test_sess[0][-1]), test_img, cls_per_task=cls_per_task),
                 generate_predictions(algorithm, models[1], int(test_sess[-1][-1]), test_img, cls_per_task=cls_per_task)]

    # Move preds to cpu
    preds[0] = preds[0].cpu()
    preds[1] = preds[1].cpu()

    if preds[0].item() == preds[1].item() and preds[0].item() != test_label.item():
        print("\033[1mFound one!\033[0m")
    print(f"Sample {sample}: {preds}")

    if not subset_testing:
        # Store predictions

        # Load the saved preds, if possible
        '''
        if os.path.isfile(preds_savepath):
            loaded_preds = scipy.io.loadmat(preds_savepath, simplify_cells=True)
            keys_to_remove = ['__header__', '__version__', '__globals__']
            pred_dict = {key: value for key, value in loaded_preds.items() if key not in keys_to_remove}
        else:
            pred_dict = {}

        # Fix ds-al name formatting for saving
        if algorithm == "ds-al":
            algorithm = "dsal"

        if f'{algorithm}' not in pred_dict: pred_dict[f'{algorithm}'] = {}
        if f'sample{sample}' not in pred_dict[f'{algorithm}']: pred_dict[f'{algorithm}'][f'sample{sample}'] = {}
        pred_dict[f'{algorithm}'][f'sample{sample}'][f'pred_{test_sess[0]}'] = preds[0].item()
        pred_dict[f'{algorithm}'][f'sample{sample}'][f'pred_{test_sess[-1]}'] = preds[1].item()
        '''

        # Save shap values to filepath
        #scipy.io.savemat(preds_savepath, pred_dict)
