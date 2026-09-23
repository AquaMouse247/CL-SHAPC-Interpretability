function [threshold] = getshapcthreshold(alg, dataset, subset_testing, subset_num)
%getshapcthreshold(alg, dataset) Computes the high/low SHAPC threshold for
% a given algorithm and dataset.
%   Detailed explanation goes here
arguments (Input)
    alg string
    dataset string
    subset_testing logical = false
    subset_num double = 0
end

arguments (Output)
    threshold
end

% Select correct config and extract parameters
setup;
config = dataset_configs.(dataset);
num_sessions = config.num_sessions;

if subset_testing & ~subset_num==0
    if subset_num==0
        % Load SHAPC values
        if strcmp(dataset, "cifar100")
            save_path = sprintf("%s/%s/shapc_vals_first_last_2000.mat", alg, dataset);
        elseif strcmp(dataset, "imagenet200")
            save_path = sprintf("%s/%s/shapc_vals_first_last_4000.mat", alg, dataset);
        else
            save_path = sprintf("%s/%s/shapc_vals_first_last_1000.mat", alg, dataset);
        end
    else
        save_path = sprintf("subset_testing/%s/%s_shapc_vals_subset_%d.mat", dataset, alg, subset_num);
    end

else
    shapc_path = sprintf("%s_shapc_data.mat", dataset);
    
    if isfile(shapc_path)
        sh_nm = fieldnames(load(shapc_path));
        shapc_data = load(shapc_path).(string(sh_nm));
    else
        shapc_data = struct();
    end
    
    % Load SHAPC values
    if strcmp(dataset, "cifar100")
        save_path = sprintf("%s/%s/shapc_vals_first_last_2000.mat", alg, dataset);
    elseif strcmp(dataset, "imagenet200")
        save_path = sprintf("%s/%s/shapc_vals_first_last_4000.mat", alg, dataset);
    else
        save_path = sprintf("%s/%s/shapc_vals_first_last_1000.mat", alg, dataset);
    end
end


if isfile(save_path)
    shapc_struct = load(save_path);
   
    % Load SHAPC values all
    shapc_avgs = [];
    alg_shapc_vars = [];
    for i=1:num_sessions-1
        pair_str = 'sc' + string(i-1) +string(num_sessions-1);
        shapcs = [];
        sample_list = string(fieldnames(shapc_struct.(pair_str)));
        for k=1:length(fieldnames(shapc_struct.(pair_str)))
            sample_str = sample_list(k);
            shapcs = [shapcs; shapc_struct.(pair_str).(sample_str)];
        end
        shapc_avgs = [shapc_avgs; mean(shapcs)];
        rel_shapc_std = std(shapcs) / mean(shapcs);
        alg_shapc_vars = [alg_shapc_vars; rel_shapc_std];

    end
    

    shapc_mean_perc = mean(shapc_avgs);
    shapc_var_perc = mean(alg_shapc_vars) * 100;
else
    shapc_mean_perc = NaN;
    shapc_var_perc = NaN;
end

shapc_std = sqrt(shapc_var_perc);


if subset_testing & ~subset_num==0
    shapc_mean = shapc_mean_perc;
else
    if strcmp(dataset, "cifar100")
        shapc_mean = shapc_data.(alg).first_last_2000_shapc;
    elseif strcmp(dataset, "imagenet200")
        shapc_mean = shapc_data.(alg).first_last_4000_shapc;
    else
        shapc_mean = shapc_data.(alg).first_last_1000_shapc;
    end
end

threshold = shapc_mean - shapc_std;
