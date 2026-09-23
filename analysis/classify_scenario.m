function scn = classify_scenario(sample, pred1, pred2, shapc_threshold, dataset, shapc_struct)
%classify_scenario(sample, pred1, pred2, shapc_threshold) Classifies a
%image sample into one of four SHAPC-prediction scenarios.
%--------------------------------------------------------------------
% Scenarios:
% 1. Low Accuracy, Low SHAPC -> untrustworthy and inaccurate
% 2. Low Accuracy, High SHAPC -> trustworthy but inaccurate
% 3. High Accuracy, Low SHAPC -> accurate but untrustworthy
% 4. High Accuracy, High SHAPC -> ideal model, accurate and trustworthy
%--------------------------------------------------------------------
arguments (Input)
    sample
    pred1
    pred2
    shapc_threshold
    dataset
    shapc_struct
end

arguments (Output)
    scn
end

% Select correct config and extract parameters
setup;
config = dataset_configs.(dataset);
num_sessions = config.num_sessions;
num_classes = config.num_classes;
cls_per_task = config.cls_per_task;
samples_per_cls = config.samples_per_cls;

true_label = floorDiv(str2double(sample), samples_per_cls);
%print("Label:", true_label)
task = floorDiv(true_label, cls_per_task);

if strcmp(dataset, 'cifar100') || strcmp(dataset, 'imagenet200')
    pair_str = 'sc' + string(task) + '9';
else
    pair_str = 'sc' + string(task) + '4';
end

sample_str = 'sample' + string(sample);
shapc = shapc_struct.(pair_str).(sample_str);

if pred1 ~= pred2
    % Scenario 1 or 2
    if shapc < shapc_threshold
      scn = 1;
    else
      scn = 2;
    end
elseif pred1 == pred2
    % Scenario 2, 3, or 4
    if pred1 ~= true_label
      scn = 2;
    elseif shapc < shapc_threshold
      scn = 3;
    else
      scn = 4;
    end
end
end


  