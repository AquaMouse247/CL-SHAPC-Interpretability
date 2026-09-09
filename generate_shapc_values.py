import numpy as np
from tqdm import tqdm
import scipy.io

from generate_shap_values import subset_testing, subset_num
from utils.setup_args import SHAPArgs, create_shap_value_filepath, create_shapc_savepath


def normalize_shap_value(shap_val):
    # If the input was (1, C, H, W), the output will be (1, C, H, W).
    # We want (H, W, C) for a single image for simpler indexing later.
    shap_val.squeeze()
    if shap_val.ndim == 4 and shap_val.shape[0] == 1:
        shap_val = np.transpose(shap_val[0], (1, 2, 0))  # From (C, H, W) to (H, W, C)
    elif shap_val.ndim == 3: #and shap_val.shape[0] == background_data.shape[1]:  # This means it's (C, H, W)
        shap_val = np.transpose(shap_val, (1, 2, 0))  # From (C, H, W) to (H, W, C)
    # Handle grayscale if C=1, imshow expects (H,W) or (H,W,1)
    if shap_val.shape[-1] == 1:
        shap_val = shap_val.squeeze(-1)  # Convert (H,W,1) to (H,W)

    # Perform Min-Max Normalization as described in the paper
    # This should be applied per sample and channel for consistency.
    # The paper says "on si,j (ft, x)", implying for the specific sample's features.
    # Let's normalize to [0, 1] based on its own min/max, per channel if multichannel
    shap_val_normalized = np.zeros_like(shap_val, dtype=np.float32)
    if shap_val.ndim == 2:  # Grayscale (H, W)
        min_val, max_val = shap_val.min(), shap_val.max()
        if max_val - min_val > 1e-6:  # Avoid division by zero
            normalized_shap_val_np = (shap_val - min_val) / (max_val - min_val)
        else:
            normalized_shap_val_np = np.zeros_like(shap_val)  # All values same, map to 0
    else:  # Multi-channel (H, W, C)
        for c in range(shap_val.shape[-1]):
            channel_data = shap_val[..., c]
            min_val, max_val = channel_data.min(), channel_data.max()
            if max_val - min_val > 1e-6:
                shap_val_normalized[..., c] = (channel_data - min_val) / (max_val - min_val)
            else:
                shap_val_normalized[..., c] = np.zeros_like(channel_data)

    return shap_val_normalized # Returns (H, W, C) or (H, W) for grayscale


# --- 2. Function to compute the Important Feature Area (pt(x)) mask ---
def get_important_feature_area(shap_values_norm, threshold_strategy='percentile', percentile=70):
    """
    Computes the important feature area based on normalized SHAP values.

    Args:
        shap_values_norm (np.ndarray): Normalized SHAP values for a sample, (H, W, C) or (H, W).
        threshold_strategy (str): 'percentile' or 'absolute'.
        percentile (int): If strategy is 'percentile', the percentile to use (e.g., 90 for top 10%).

    Returns:
        np.ndarray: Binary mask of important features, same shape as shap_values_norm.
                    1 where SHAP value >= threshold, 0 otherwise.
    """
    if threshold_strategy == 'percentile':
        # Flatten and get the threshold based on all SHAP values for the sample
        # Or you might want to consider magnitude: np.percentile(np.abs(shap_values_norm), percentile)
        # The paper implies si,j(f,x) >= sTh, not |si,j| >= sTh. So, use raw values.
        s_th = np.percentile(shap_values_norm, percentile)
    elif threshold_strategy == 'absolute':
        # You'd define an absolute threshold value here
        s_th = 0.5  # Example absolute threshold, adjust as needed
    else:
        raise ValueError("threshold_strategy must be 'percentile' or 'absolute'")

    # Create the binary mask
    mask = (shap_values_norm >= s_th).astype(int)
    return mask


# --- 3. Function to calculate SHAP Value Consistency (SHAPC) ---
def calculate_shapc(shap_tau_norm, p_tau, shap_t_norm, p_t):
    """
    Calculates the SHAP Value Consistency (SHAPC) for a single sample x
    between two tasks (tau and t).

    Args:
        shap_tau_norm (np.ndarray): Normalized SHAP values for task tau, (H, W, C) or (H, W).
        p_tau (np.ndarray): Important feature mask for task tau, (H, W, C) or (H, W).
        shap_t_norm (np.ndarray): Normalized SHAP values for task t, (H, W, C) or (H, W).
        p_t (np.ndarray): Important feature mask for task t, (H, W, C) or (H, W).

    Returns:
        float: The SHAPC value for the sample, averaged across channels if multi-channel.
    """

    # Ensure all inputs have the same shape
    if not (shap_tau_norm.shape == p_tau.shape == shap_t_norm.shape == p_t.shape):
        raise ValueError("All SHAP value arrays and masks must have the same shape.")

    # Convert to single channel if multi-channel for the core calculation, then average
    ### Double-check this? ###
    if shap_tau_norm.ndim == 3:  # Multi-channel (H, W, C)
        num_channels = shap_tau_norm.shape[-1]
        channel_shapc_values = []
        for c in range(num_channels):
            s_tau_c = shap_tau_norm[..., c]
            m_tau_c = p_tau[..., c]
            s_t_c = shap_t_norm[..., c]
            m_t_c = p_t[..., c]
            channel_shapc_values.append(_calculate_single_channel_shapc(s_tau_c, m_tau_c, s_t_c, m_t_c))
        return np.mean(channel_shapc_values)
    else:  # Grayscale (H, W)
        return _calculate_single_channel_shapc(shap_tau_norm, p_tau, shap_t_norm, p_t)


def _calculate_single_channel_shapc(s_tau, m_tau, s_t, m_t):

    # Intersection of important feature areas for numerator part
    intersection_mask = (m_tau == 1) & (m_t == 1)

    # Union of important feature areas for denominator
    union_mask = (m_tau == 1) | (m_t == 1)

    # Calculate the e^(-|diff|) term
    exp_diff = np.exp(-np.abs(s_t - s_tau))

    numerator_sum = np.sum(exp_diff[intersection_mask])
    denominator_sum = np.sum(exp_diff[union_mask])

    if denominator_sum == 0:
        return 0.0  # Avoid division by zero if no important features in union
    else:
        return (numerator_sum / denominator_sum) * 100


if __name__ == "__main__":

    #algorithms = ["iTAML", "RPSnet", "DGR", "foster", "memo", "der", "icarl", "dsal", "tagfex", "xder"]
    algorithms = ["der"]
    dataset = "cifar10"


    if len(algorithms) > 1 and dataset != "mnist" and "DGR" in algorithms:
        algorithms.remove("DGR")

    for alg in algorithms:
        algorithm = alg
        shapArgs = SHAPArgs(algorithm, dataset)

        num_sessions = shapArgs.dataset_params.num_task
        cls_per_task = shapArgs.dataset_params.class_per_task
        init_cls = shapArgs.dataset_params.init_cls

        first_last_only = True
        all_samples = False
        subset_testing = True
        subset_num = 0
        filepath = create_shap_value_filepath(shapArgs, first_last_only, subset_testing, subset_num) + ".npy"
        savepath = create_shapc_savepath(shapArgs, first_last_only, all_samples, subset_testing, subset_num)


        # Load the SHAP Values
        shap_values_loaded = np.load(filepath, allow_pickle=True)  # ['shap_dict']
        num_imgs = len(shap_values_loaded[()].keys())
        shap_dict = {}
        for i in range(num_imgs):
            shap_dict[f'{i}'] = shap_values_loaded[()][f'{i}']

        shapc_dict = {}
        for sample in tqdm(range(num_imgs), desc="Progress"):

            if shap_dict[f'{sample}']['true_label'] < init_cls:
                start_sess = 0
            else:
                start_sess = ((shap_dict[f'{sample}']['true_label'] - init_cls) // cls_per_task) + 1

            # If the sample is from the last task skip
            if start_sess >= num_sessions-1: continue

            if first_last_only: range1 = [start_sess]
            else: range1 = range(start_sess, num_sessions-1)

            for ses in range1:
                shap_value1 = shap_dict[f'{sample}'][f'ses{ses}']['shap_values']
                if dataset == "mnist" and algorithm == "RPSnet":
                    shap_value1 = shap_value1.reshape(28,28,1)
                    shap_value1 = np.expand_dims(shap_value1, 0)
                    shap_value1 = np.expand_dims(shap_value1, 0)
                shap_value1 = normalize_shap_value(shap_value1)

                if first_last_only:
                    if algorithm == "iTAML": range2 = [num_sessions-1+ses]
                    else: range2 = [num_sessions-1]
                else: range2 = range(ses+1, num_sessions)

                for j in range2:

                    shap_value2 = shap_dict[f'{sample}'][f'ses{j}']['shap_values']
                    if dataset == "mnist" and algorithm == "RPSnet":
                        shap_value2 = shap_value2.reshape(28, 28, 1)
                        shap_value2 = np.expand_dims(shap_value2, 0)
                        shap_value2 = np.expand_dims(shap_value2, 0)
                    shap_value2 = normalize_shap_value(shap_value2)


                    # Compute Important Feature Area Masks
                    # Using percentile threshold, top 30% most important features
                    p_tau = get_important_feature_area(shap_value1, percentile=70)
                    p_t = get_important_feature_area(shap_value2, percentile=70)


                    # Calculate SHAPC
                    shapc_value = calculate_shapc(shap_value1, p_tau, shap_value2, p_t)
                    last_task = j-ses if first_last_only and algorithm == "iTAML" else j
                    if f'sc{ses}{last_task}' not in shapc_dict: shapc_dict[f'sc{ses}{last_task}'] = {}
                    shapc_dict[f'sc{ses}{last_task}'][f'sample{sample}'] = shapc_value

        scipy.io.savemat(savepath, shapc_dict)