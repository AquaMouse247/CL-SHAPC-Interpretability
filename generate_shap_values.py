import numpy as np
import shap
import torch
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm

# Custom Imports
from utils.setup_args import SHAPArgs, create_shap_value_filepath, get_subset_filepath
from utils.load_models import load_model, load_meta_models
from utils.model_parameters import pycil_algs

import utils.shap_dataloader as sdl
from models.RPSnet.rps_net import generate_path



algorithm = "foster"
dataset = "cifar10"
shapArgs = SHAPArgs(algorithm, dataset)

'''
# Used when running with a shell script.
algorithm = sys.argv[1]
dataset = sys.argv[2]
#'''

first_last_only = True
subset_testing = True
subset_num = 1
filepath = create_shap_value_filepath(shapArgs, first_last_only, subset_testing, subset_num)

print(f"Alg: {algorithm}\nDataset: {dataset}\nFirst/Last: {first_last_only}")
num_tasks = shapArgs.dataset_params.num_task
cls_per_task = shapArgs.dataset_params.class_per_task

# Configures models depending on the algorithm chosen
if algorithm == "iTAML":
    models = []
    for i in range(num_tasks):
        models.extend(load_meta_models(dataset, i))
    for model in models: model.set_saliency(True)

else:
    models = [load_model(algorithm, dataset, i, shapArgs=shapArgs) for i in range(num_tasks)]
    if algorithm == "RPSnet":
        infer_paths = []
        lasts = [-1] * len(models)
        for i in range(len(models)):
            infer_paths.append(generate_path(i, dataset, shapArgs.algorithm_args))
            models[i].shap_path = infer_paths[i]
            models[i].set_shap(True)
    elif algorithm in pycil_algs:
        for model in models: model.set_shap(True)
    elif algorithm == "xder":
        #for model in models: model.set_shap(True)
        pass


sal_dataloader = sdl.ShapDataloader(shapArgs)

# Get train dataset
train_set = sal_dataloader.get_shap_train_set(dataset)

if subset_testing:
    shap_subset = torch.load(get_subset_filepath(shapArgs), weights_only=False)[subset_num-1]
    shap_dataloader = DataLoader(shap_subset, batch_size=10000, shuffle=False)
    *_, test_imgs, test_labels = next(iter(shap_dataloader))
    print("Loaded Imgs:", len(test_imgs))
else:
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

# Reshape MNIST test images for RPSnet
if algorithm == "RPSnet" and dataset == "mnist":
    test_imgs = test_imgs.detach().numpy().reshape(-1, 784)
    test_imgs = torch.from_numpy(test_imgs)


# 1. Generate 100 random indices
indices = np.random.choice(len(train_set), 100, replace=False)

# 2. Create a subset of the dataset
small_train_set = Subset(train_set, indices)

# Get background data for shap explainer
if algorithm == "RPSnet" and dataset == "mnist":
    background_data = torch.cat([x[0].unsqueeze(0).reshape(-1, 784) for x in small_train_set])  # Getting a sample
else:
    background_data = torch.cat([img.unsqueeze(0) for (*_, img, lbl) in small_train_set])  # Getting a sample
background_data = shap.sample(background_data, 100) # Taking 100 random samples

#Create the explainer for each model
explainers = [shap.GradientExplainer(models[i], background_data) for i in range(len(models))]

shap_dict = {}
all_shap_values = []
all_indexes = []
sample = 0

for sample in tqdm(range(len(test_imgs)), desc="Progress"):
    shap_dict[f'{sample}'] = {}
    sample_task = test_labels[sample] // cls_per_task
    for e in range(len(explainers)):
        if first_last_only:
            if algorithm == "iTAML": boolean_statement = (e == sample_task or e == num_tasks-1+sample_task)
            else: boolean_statement = (e == sample_task or e == num_tasks-1)
        else:
            boolean_statement = test_labels[sample] // cls_per_task <= e

        if boolean_statement:
            #print("Shape:", test_imgs.shape)
            #print("Sample Shape:", test_imgs[sample].unsqueeze(0).shape)
            shap_values, indexes = explainers[e].shap_values(test_imgs[sample].unsqueeze(0), ranked_outputs=1)
            shap_dict[f'{sample}'][f'ses{e}'] = {'shap_values': shap_values, 'idxs': indexes}
            shap_dict[f'{sample}']['true_label'] = test_labels[sample].item()
        else:
            continue


    # Intermittent saving in case of crash
    #np.save(filepath, shap_dict)

# Final save shap values to filepath
np.save(filepath, shap_dict)
