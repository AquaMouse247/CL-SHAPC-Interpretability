import sys

import numpy as np
import shap
import torch
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm

# Custom Imports
from utils.setup_args import SHAPArgs, create_shap_value_filepath, get_subset_filepath
from utils.load_models import load_model, load_meta_models
from utils.model_parameters import pycil_algs
from utils.datasets import Bootstrap

import utils.shap_dataloader as sdl
from models.RPSnet.rps_net import generate_path

device = "cuda:0" if torch.cuda.is_available() else "cpu"

if len(sys.argv[1:]) > 2:
    # Used when running with a shell script.
    algorithm = sys.argv[1]
    dataset = sys.argv[2]
    subset_testing = int(sys.argv[3])
    subset_num = int(sys.argv[4])
else:
    algorithm = "foster"
    dataset = "cifar10"
    subset_testing = True
    subset_num = 101


shapArgs = SHAPArgs(algorithm, dataset)

first_last_only = True

filepath = create_shap_value_filepath(shapArgs, first_last_only, subset_testing, subset_num)

print(f"Alg: {algorithm}\nDataset: {dataset}\nFirst/Last: {first_last_only}")
print(f"Subset Testing: {subset_testing}\nSubset Num: {subset_num}")
num_tasks = shapArgs.dataset_params.num_task
cls_per_task = shapArgs.dataset_params.class_per_task

# Configures models depending on the algorithm chosen
if algorithm == "iTAML":
    models = []
    for i in range(num_tasks):
        models.extend(load_meta_models(dataset, i))
    for model in models:
        model.set_saliency(True)
        model.to(device)

else:
    models = [load_model(algorithm, dataset, i, shapArgs=shapArgs).to(device) for i in range(num_tasks)]
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

test_imgs, test_labels = test_imgs.to(device), test_labels.to(device)

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
background_data = background_data.to(device)

#Create the explainer for each model
explainers = [shap.GradientExplainer(models[i], background_data) for i in range(len(models))]

shap_dict = {}
all_shap_values = []
all_indexes = []
sample = 0

batch_size=32
img_ses_idxs = []
[img_ses_idxs.append([]) for i in range(num_tasks)]

for sample in range(len(test_imgs)):
    shap_dict[f'{sample}'] = {}
    sample_task = test_labels[sample] // cls_per_task
    for e in range(len(explainers)):
        if first_last_only:
            if algorithm == "iTAML": boolean_statement = (e == sample_task or e == num_tasks-1+sample_task)
            else: boolean_statement = (e == sample_task or e == num_tasks-1)
        else:
            boolean_statement = test_labels[sample] // cls_per_task <= e

        if boolean_statement:
            img_ses_idxs[e].append(sample)
        else:
            continue


for e in range(len(explainers)):
    print(f"Generating SHAP values for task {e} model.")

    for batch_start in tqdm(range(0, len(img_ses_idxs[e]), batch_size), desc="Progress"):
        batch_idxs = img_ses_idxs[e][batch_start:batch_start + batch_size]
        #batch_inputs = test_imgs[batch_idxs]
        batch_inputs = test_imgs[batch_idxs]

        # Calculate SHAP values for the entire mini-batch at once
        shap_values, indexes = explainers[e].shap_values(batch_inputs, ranked_outputs=1)
        #shap_dict[f'{sample}'][f'ses{e}'] = {'shap_values': shap_values, 'idxs': indexes}
        #shap_dict[f'{sample}']['true_label'] = test_labels[sample].item()

        for batch_idx, subset_idx in enumerate(batch_idxs):
            val, idx_val = shap_values[batch_idx], indexes[batch_idx]
            shap_dict[f'{subset_idx}'][f'ses{e}'] = {"shap_values": val, "idxs": idx_val}

'''
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
'''

    # Intermittent saving in case of crash
    #np.save(filepath, shap_dict)

# Final save shap values to filepath
np.save(filepath, shap_dict)





# Gemini Optimized Version
'''

Conversation with Gemini
Temporary Chat
Is there any way to run this faster? Specifically, if I plan on using Google Colab:

Yes. The primary bottleneck is sample-by-sample computation coupled with underutilizing Colab's GPU. shap.GradientExplainer computes forward and backward passes for every evaluation instance combined with the background dataset. Running this sequentially with an inner model loop causes massive CPU-GPU synchronization overhead.

Key Optimizations
1. Move Models and Data Explicitly to GPU
Ensure your models and tensors are explicitly placed on CUDA. In PyTorch, SHAP's GradientExplainer will run on GPU if the model and input tensors are on CUDA.

Python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for model in models:
    model.to(device)
    model.eval()

background_data = background_data.to(device)
test_imgs = test_imgs.to(device)
2. Batch Test Inputs instead of Looping Single Samples
Currently, you compute SHAP one image at a time (test_imgs[sample].unsqueeze(0)). shap_values() natively accepts batched inputs (e.g., batch_size=32 or 64), which drastically reduces kernel launch and loop overhead.

Filter your test indices per explainer upfront, run mini-batches through explainers[e].shap_values(batch), and map them back to your dictionary.

3. Reduce or Optimize the Background Distribution
GradientExplainer integrates gradients between each test sample and each background sample.

Computing for 1 image against 100 background samples requires 100 gradient evaluations per sample.

Reducing background_data from 100 to 30–50 samples (e.g., using shap.kmeans or simply shap.sample(..., 30)) yields an immediate 2–3x speedup with minimal impact on attribution accuracy.

4. Instantiate Explainers Lazily
Currently, you initialize all explainers = [shap.GradientExplainer(...) for i in range(len(models))] upfront. Because first_last_only = True, many models might never be queried, or you only query 2 models per sample (sample_task and the last task). Only instantiate and hold the active explainers on GPU to avoid VRAM bloat and initialization lag.

Optimized Replacement for the Evaluation Loop
Here is a refactored version of the evaluation step grouping samples into mini-batches:

Python
# 1. Setup Device & Background
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for m in models:
    m.to(device).eval()

# Reduce background size slightly if speed is critical (e.g., 40-50 samples)
background_data = background_data.to(device)
test_imgs = test_imgs.to(device)
test_labels = test_labels.to(device)

# 2. Pre-filter samples relevant to each explainer to batch operations
BATCH_SIZE = 32  # Tune based on Colab GPU VRAM (16, 32, or 64)
shap_dict = {f"{i}": {"true_label": test_labels[i].item()} for i in range(len(test_imgs))}

# Determine which explainer indices are required overall
required_explainers = set()
for sample in range(len(test_imgs)):
    sample_task = (test_labels[sample] // cls_per_task).item()
    if first_last_only:
        if algorithm == "iTAML":
            required_explainers.update([sample_task, num_tasks - 1 + sample_task])
        else:
            required_explainers.update([sample_task, num_tasks - 1])
    else:
        required_explainers.update([e for e in range(len(models)) if sample_task <= e])

# 3. Process Explainer by Explainer (maximizes GPU cache and avoids switching overhead)
for e in sorted(required_explainers):
    if e >= len(models):
        continue

    explainer = shap.GradientExplainer(models[e], background_data)

    # Collect indices that need evaluation under model e
    matching_indices = []
    for sample in range(len(test_imgs)):
        sample_task = (test_labels[sample] // cls_per_task).item()
        if first_last_only:
            cond = (e == sample_task or e == (num_tasks - 1 + sample_task if algorithm == "iTAML" else num_tasks - 1))
        else:
            cond = (sample_task <= e)
        if cond:
            matching_indices.append(sample)

    # Batch compute SHAP values
    for b_start in tqdm(range(0, len(matching_indices), BATCH_SIZE), desc=f"Explainer {e}"):
        b_indices = matching_indices[b_start:b_start + BATCH_SIZE]
        batch_inputs = test_imgs[b_indices]

        # Calculate SHAP values for the entire mini-batch at once
        shap_vals, idxs = explainer.shap_values(batch_inputs, ranked_outputs=1)

        # Distribute batch outputs back to dictionary
        for local_idx, orig_idx in enumerate(b_indices):
            # GradientExplainer outputs lists per class output or arrays
            val = shap_vals[local_idx] if isinstance(shap_vals, np.ndarray) else [sv[local_idx] for sv in shap_vals]
            idx_val = idxs[local_idx] if hasattr(idxs, "__getitem__") else idxs
            
            shap_dict[f"{orig_idx}"][f"ses{e}"] = {
                "shap_values": val,
                "idxs": idx_val
            }

# Final save
np.save(filepath, shap_dict)

PyTorch Inference Mode: Wrap your explainer calls with torch.inference_mode(False) or ensure gradients are only calculated for inputs, disabling unnecessary weight gradients (param.requires_grad = False for all model weights).

'''