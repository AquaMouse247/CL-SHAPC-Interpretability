### Load compare dict
from fileinput import filename

import numpy as np
import os
import re
import scipy
import textwrap
import addcopyfighandler

from utils.model_parameters import pycil_algs
# Custom Imports
from utils.setup_args import SHAPArgs, create_shap_value_filepath, create_preds_savepath
import utils.shap_dataloader as sdl


def format_name(name):
    if name == "icarl":
        return "iCARL"
    elif name == "tagfex":
        return "TagFex"
    elif name == "dsal":
        return "DS-AL"
    elif name == "xder":
        return "X-DER"
    elif name in pycil_algs or name == "xder":
        return name.upper()
    else:
        return name


def add_algorithm_labels(fig, axis, algs):

    fig.canvas.draw()

    # Offsets in figure coordinate space (0 to 1)
    y_offset_top = 0.08  # Upper tier (further above)

    # Pairs of columns to place text between: (0, 1), (2, 3), (4, 5)
    for i, col_left in enumerate([0, 2, 4]):
        ax_left = axis[0, col_left]
        ax_right = axis[0, col_left + 1]

        # Get bounding boxes in figure coordinates (0 to 1)
        bbox_left = ax_left.get_position()
        bbox_right = ax_right.get_position()

        # Midpoint horizontally between the two subplots
        x_mid = (bbox_left.x1 + bbox_right.x0) / 2.0

        fig.text(x_mid, bbox_left.y1 + y_offset_top, format_name(algs[i]),
                ha="center",va="bottom", fontsize=14,
                fontweight="bold")

def get_pred_labels(data_name, data_classes, alg_name, sample, mat_file=False):
    if mat_file:
        filename = f"analysis/preds/{data_name}_preds.mat"

        if not os.path.isfile(filename):
            print(f"Could not find preds file at '{filename}'.")
            return
        else:
            loaded_preds = scipy.io.loadmat(filename, simplify_cells=True)
            keys_to_remove = ['__header__', '__version__', '__globals__']
            pred_dict = {key: value for key, value in loaded_preds.items() if key not in keys_to_remove}

            test_sess = list(pred_dict['preds_data'][f'{alg_name}'][f'sample{sample}'].keys())
            pred1 = pred_dict['preds_data'][f'{alg_name}'][f'sample{sample}'][test_sess[0]]
            pred2 = pred_dict['preds_data'][f'{alg_name}'][f'sample{sample}'][test_sess[1]]

            pred1, pred2 = data_classes[int(pred1)], data_classes[int(pred2)]

    else:
        filename = f"analysis/preds/{alg_name}_{data_name}_preds.out"

        if not os.path.exists(filename):
            print(f"Could not find preds file at '{filename}'.")
            return

        with open(filename, "r") as f:
            lines = f.readlines()

        for line in lines:
            if "Sample" not in line:
                continue

            parts = line.split(":")

            tokens = re.findall(r"\d+", parts[0])
            file_sample = tokens[0]
            if not int(file_sample) == sample:
                continue

            prediction = re.findall(r"\d+", parts[1])
            pred1 = data_classes[int(prediction[0])]
            pred2 = data_classes[int(prediction[1])]

    return [pred1, pred2]



# Select Algorithm and Dataset
algorithm = "RPSnet"
dataset = "cifar100"

vis_all_algs = True
all_algs = ["iTAML", "RPSnet", "foster", "memo", "der", "icarl", "tagfex", "dsal", "xder"]
if vis_all_algs:
    alg_list = all_algs
else:
    alg_list = [algorithm]


first_last_only = True


import matplotlib.pyplot as plt
# Prepare figs
figs = []
current_algs = []


for a, alg in enumerate(alg_list):
    if vis_all_algs:
        if a % 3 == 0:
            plt_fig, plt_axis = plt.subplots(2, 3 * 2, figsize=(3 * 4, 4))
            current_algs = []
    else:
        plt_fig, plt_axis = plt.subplots(2, 2, figsize=(6, 6))

    current_algs.append(alg)
    shapArgs = SHAPArgs(alg, dataset)

    num_tasks = shapArgs.dataset_params.num_task
    num_class = shapArgs.dataset_params.num_class
    cls_per_task = shapArgs.dataset_params.class_per_task
    shap_samples = shapArgs.dataset_params.shap_samples

    print("Current Alg:", alg)
    filepath = create_shap_value_filepath(shapArgs, first_last_only) + ".npy"
    #print("Current Filepath:", filepath)

    shap_values_loaded = np.load(filepath, allow_pickle=True)  # ['shap_dict']
    #shap_values_loaded = np.load(f"analysis/{algorithm}/{dataset}/shap_values_first_last_1000.npy", allow_pickle=True)  # ['shap_dict']
    num_imgs = len(shap_values_loaded[()].keys())
    shap_dict = {}
    for i in range(num_imgs):
        shap_dict[f'{i}'] = shap_values_loaded[()][f'{i}']

    '''
    Bad Sample: 80
    Good Sample: 132
    
    Samples to Try: 55
    '''

    sample = 953
    test_sample = shap_dict[f'{sample}']
    test_sess = list(test_sample.keys())
    test_sess.remove(test_sess[1])
    print(test_sess)
    ses, last_ses = int(test_sess[0].replace('ses', '')), int(test_sess[-1].replace('ses', ''))

    '''
    if dataset == "mnist":
        test_shaps = [shap_dict[f'{sample}'][f'ses{ses}']['shap_values'].reshape(28,28,1),
                      shap_dict[f'{sample}']['ses4']['shap_values'].reshape(28,28,1)]
    else:
        test_shaps = [shap_dict[f'{sample}'][f'ses{ses}']['shap_values'].squeeze().reshape(32,32,3),
                      shap_dict[f'{sample}']['ses4']['shap_values'].squeeze().reshape(32,32,3)]
        print(test_shaps[0].shape)
    '''

    # Reshape/Transpose to Batch x Height x Width x Channel
    if dataset == "mnist":
        test_shaps = [shap_dict[f'{sample}'][f'ses{ses}']['shap_values'].reshape(28,28,1),
                      shap_dict[f'{sample}'][f'ses{last_ses}']['shap_values'].reshape(28,28,1)]
    else:
        test_shaps = [shap_dict[f'{sample}'][f'ses{ses}']['shap_values'].squeeze(-1).transpose([0, 2, 3, 1]),
                      shap_dict[f'{sample}'][f'ses{last_ses}']['shap_values'].squeeze(-1).transpose([0, 2, 3, 1])]

    # Get test dataset
    sal_dataloader = sdl.ShapDataloader(shapArgs)

    if dataset == "cifar100":
        sal_imgs, sal_labels, sal_classes, STD, MEAN = sal_dataloader.load_data(range(ses * 10, (ses * 10) + 10), 20, batch_size=10000)
    elif dataset == "imagenet200":
        sal_imgs, sal_labels, sal_classes, STD, MEAN = sal_dataloader.load_data(range(ses * 20, (ses * 20) + 20), 20, batch_size=10000)
    else:
        ###---Updated to take initial classes learned into account---###
        if ses == 0:
            sal_imgs, sal_labels, sal_classes, STD, MEAN = sal_dataloader.load_data(range(shapArgs.dataset_params.init_cls),
                                                                          shapArgs.dataset_params.shap_samples, batch_size=10000)
        else:
            sal_imgs, sal_labels, sal_classes, STD, MEAN = sal_dataloader.load_data(range((shapArgs.dataset_params.init_cls-cls_per_task)+ses*cls_per_task,
                                                                            shapArgs.dataset_params.init_cls+(ses*cls_per_task)),
                                                                            shapArgs.dataset_params.shap_samples, batch_size=10000)
        ###----------------------------------------------------------###
    #print("Len of sal_imgs:", len(sal_imgs))

    test_imgs, test_labels, test_classes = sal_imgs, sal_labels, sal_classes

    # Get test image
    samples = range(shap_samples*(num_class-cls_per_task))

    adj_sample = sample - (ses * cls_per_task * shap_samples)

    test_img = test_imgs[adj_sample]#.unsqueeze(0)
    test_label = test_labels[adj_sample]
    if dataset != "mnist":
        test_img = sal_dataloader.denormalize(test_img)
    test_img_np = np.transpose(test_img.numpy(), [1, 2, 0])
    #print(f"Test Label:", test_classes[test_label])

    labels = [f'ses{ses}', f'ses{num_tasks-1}']
    use_mat_file = True if dataset == "cifar10" else False
    preds = get_pred_labels(dataset, test_classes, alg, sample, mat_file=use_mat_file)
    #shap.image_plot(np.concatenate(test_shaps), np.stack([test_img_np,test_img_np]), true_labels=labels, cmap='plasma')


    # For a model with high SHAPC, but low accuracy:
    # Look for a sample that was predicted correctly, feature consistency should be high
    # Look for a sample that was predicted incorrectly, but check if feature consistency is still high -> indicates trustworthiness

    alg_col_index = (a - (a//3*3)) * 2

    #print("Row:",alg_row_index)
    #print("Col:", alg_col_index)

    for ax in plt_axis[:, alg_col_index]:
        ax.imshow(test_img_np)
        ax.set_title("Original Image")
        label_str = f"True Label: {test_classes[test_label]}"
        if alg_col_index == 0: ax.set_xlabel("\n".join(textwrap.wrap(label_str, width=12)), fontsize=10)
    for ax in plt_axis[:, alg_col_index+1]:
        ax.imshow(np.mean(test_img_np, axis=2), cmap="gray")

    s0 = np.sum(np.abs(test_shaps[0].squeeze()), axis=-1)
    s1 = np.sum(np.abs(test_shaps[1].squeeze()), axis=-1)

    # Normalize to [0, 1] across both or let imshow scale automatically
    vmax = max(s0.max(), s1.max())
    vmin = min(s0.min(), s1.min())

    hm1 = plt_axis[0, alg_col_index+1].imshow(s0, cmap='plasma', vmin=vmin, vmax=vmax, alpha=0.5)
    plt_axis[0, alg_col_index+1].set_title(labels[0])
    plt_axis[0, alg_col_index + 1].set_xlabel(f"Pred: {preds[0]}")

    hm2 = plt_axis[1, alg_col_index+1].imshow(s1, cmap='plasma', vmin=vmin, vmax=vmax, alpha=0.5)
    plt_axis[1, alg_col_index+1].set_title(labels[1])
    plt_axis[1, alg_col_index + 1].set_xlabel(f"Pred: {preds[1]}")

    for ax in plt_axis.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    #plt.subplots_adjust(right=0.82)
    #plt_fig.subplots_adjust(top=0.82)
    #plt_fig.suptitle(f"{alg}")
    #plt.show()

    if a in [2, 5, 8]:

        cbar = plt_fig.colorbar(hm2, ax=plt_axis.ravel().tolist(), orientation='vertical', pad=0.04)
        cbar.ax.set_title("Most Important", pad=8, size=12)
        cbar.ax.set_xlabel("Least Important", labelpad=8, size=12)

        # ADD Algorithm Labels
        add_algorithm_labels(plt_fig, plt_axis, current_algs)

        #figs.append(copy.deepcopy(plt_fig))


#print(figs)
#for f in figs:
#    f.show(blocking=True)
#    pass
if not vis_all_algs:
    cbar = plt_fig.colorbar(hm2, ax=plt_axis.ravel().tolist(), orientation='vertical', pad=0.04)
    cbar.ax.set_title("Most Important", pad=8, size=12)
    cbar.ax.set_xlabel("Least Important", labelpad=8, size=12)
    plt.suptitle(format_name(algorithm), fontsize=14, fontweight="bold")

plt.show()
