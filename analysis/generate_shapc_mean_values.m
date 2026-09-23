% SHAPC-Mean and SHAPC-Var Comparison
setup; % Loads alg list and dataset_configs
       % If not all algs are needed either
       % change the algs listed in setup.m
       % OR uncomment the below line of code

%algs = ["iTAML"];       

dataset = 'imagenet200';
config = dataset_configs.(dataset);
num_sessions = config.num_sessions;
num_classes = config.num_classes;
cls_per_task = config.cls_per_task;
samples_per_cls = config.samples_per_cls;

shapc_path = sprintf("%s_shapc_data.mat", dataset);

if isfile(shapc_path)
    sh_nm = fieldnames(load(shapc_path));
    shapc_data = load(shapc_path).(string(sh_nm));
else
    shapc_data = struct();
end
%%
% Load SHAPC values (First and Last 1000)
for i=1:length(algs)
    alg = algs(i);
    if strcmp(dataset, "cifar100")
        save_path = sprintf("%s/%s/shapc_vals_first_last_2000.mat", alg, dataset);
    elseif strcmp(dataset, "imagenet200")
        save_path = sprintf("%s/%s/shapc_vals_first_last_4000.mat", alg, dataset);
    else
        save_path = sprintf("%s/%s/shapc_vals_first_last_1000.mat", alg, dataset);
    end

    if isfile(save_path)
        shapc_struct = load(save_path);
       
        % Load SHAPC values all
        shapc_avgs = [];
        for i=1:num_sessions-1
            pair_str = 'sc' + string(i-1) +string(num_sessions-1);
            shapcs = [];
            sample_list = string(fieldnames(shapc_struct.(pair_str)));
            for k=1:length(fieldnames(shapc_struct.(pair_str)))
                sample_str = sample_list(k);
                shapcs = [shapcs; shapc_struct.(pair_str).(sample_str)];
            end
            shapc_avgs = [shapc_avgs; mean(shapcs)];

        end
        
        % Note the shapc is represented as percentage
        %disp("From averaging all tasks together:")
        shapc_mean_perc = mean(shapc_avgs);
        shapc_mean = mean(shapc_avgs) / 100;
    else
        shapc_mean_perc = NaN;
        shapc_mean = NaN;
    end

    if strcmp(dataset, "cifar100")
        shapc_str = "first_last_2000_shapc";
        time_str = "first_last_2000_time";
    elseif strcmp(dataset, "imagenet200")
        shapc_str = "first_last_4000_shapc";
        time_str = "first_last_4000_time";
    else
        shapc_str = "first_last_1000_shapc";
        time_str = "first_last_1000_time";
    end
    shapc_data.(alg).(shapc_str) = shapc_mean_perc;

    %shapc_data.(alg).first_last_time = shapTimes.(dataset).first_last.(alg);
end
save(shapc_path, "shapc_data")

% Create Table
%shapc_table = table(Y', 'VariableNames', ["SHAPC-Mean"], 'RowNames', X)
%abs_diff = abs(Y(1)-Y(2))
rows = algs;
columns = ["Accuracy (%)" "SHAPC-Mean (%)" "Time (hrs)"];
column_data1 = [];
column_data2 = [];
column_data3 = [];
for i=1:length(algs)
    alg = algs(i);
    if ~isfield(shapc_data.(alg), 'acc')
        shapc_data.(alg).acc = NaN;
    end
    column_data1 = [column_data1; shapc_data.(alg).acc];
    
    column_data2 = [column_data2; shapc_data.(alg).(shapc_str)];

    if ~isfield(shapc_data.(alg), time_str)
        shapc_data.(alg).(time_str) = NaN;
    end
    column_data3 = [column_data3; shapc_data.(alg).(time_str)];
end
first_last_1000_shapcs = column_data2;
first_last_1000_times = column_data3;
shapc_table_first_last_1000 = table(column_data1, column_data2, column_data3, ...
    'VariableNames', columns, 'RowNames', rows);
sorted_shapc_first_last_1000 = sortrows(shapc_table_first_last_1000, {'SHAPC-Mean (%)'}, {'ascend'});
disp(sorted_shapc_first_last_1000) %[output:437cf833]


%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright","rightPanelPercent":40.4}
%---
%[output:437cf833]
%   data: {"dataType":"text","outputData":{"text":"              <strong>Accuracy (%)<\/strong>    <strong>SHAPC-Mean (%)<\/strong>    <strong>Time (hrs)<\/strong>\n              <strong>____________<\/strong>    <strong>______________<\/strong>    <strong>__________<\/strong>\n\n    <strong>iTAML <\/strong>         40.1           22.723          17.474  \n    <strong>xder  <\/strong>        40.05           23.432             NaN  \n    <strong>icarl <\/strong>        38.75           24.426          8.6244  \n    <strong>RPSnet<\/strong>        22.49           24.642             NaN  \n    <strong>memo  <\/strong>        55.25           28.207          20.867  \n    <strong>foster<\/strong>        54.62           28.409          18.886  \n    <strong>der   <\/strong>        57.38            37.31           52.08  \n    <strong>dsal  <\/strong>       38.248           37.589             NaN  \n    <strong>tagfex<\/strong>       57.939           37.618             NaN  \n\n","truncated":false}}
%---
