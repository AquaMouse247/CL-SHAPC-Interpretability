% Setup file
algs = ["iTAML", "RPSnet", "foster", "memo", "der", "icarl", "dsal", "tagfex", "xder"];

% Formatted names
formatted_algs = algs;
formatted_algs(3:5) = upper(algs(3:5));
formatted_algs(6) = "iCARL";
formatted_algs(7) = "DS-AL";
formatted_algs(8) = "TagFex";
formatted_algs(9) = "X-DER";

%% Dataset configs
dataset_configs = {};

% Example config:
% dataset_configs.dataset_name = struct('num_sessions', 5, 'num_classes', 10, 'cls_per_task', 2, 'samples_per_cls', 100);
% * num_sessions - total number of sessions/tasks the dataset was split into
% * num_classes - total number of classes in the dataset
% * cls_per_task - number of classes learned per session/task
% * samples_per_cls - number of samples iterated through per class during SHAP value calculation (equivalent to shap_samples)
% * init_cls, Optional - the number of classes learned in the intial class

dataset_configs.cifar10 = struct('num_sessions', 5, 'num_classes', 10, 'cls_per_task', 2, 'samples_per_cls', 100);
dataset_configs.cifar100 = struct('num_sessions', 10, 'num_classes', 100', 'cls_per_task', 10, 'samples_per_cls', 20);
dataset_configs.imagenet200 = struct('num_sessions', 10, 'num_classes', 200', 'cls_per_task', 20, 'samples_per_cls', 20);

% MedMNIST Datasets
dataset_configs.pathmnist = struct('num_sessions', 3, 'num_classes', 9, 'cls_per_task', 3, 'samples_per_cls', 100);
dataset_configs.dermamnist = struct('num_sessions', 3, 'num_classes', 7, 'cls_per_task', 2, 'init_cls', 3, 'samples_per_cls', 50);
