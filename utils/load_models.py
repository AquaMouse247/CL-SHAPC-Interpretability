import torch
import argparse
import os
import torch.optim as optim
import pickle
import numpy as np
import copy

from models.iTAML.resnet import *
from models.iTAML.radam import *

from models.iTAML import incremental_dataloader as data
from utils.model_parameters import iTAMLArgs

from utils.model_parameters import pycil_algs, get_algorithm_args
from utils.model_parameters import FOSTERNet, AdaptiveNet, DERNet, IncrementalNet, DSALNet, TagFexNet
from utils.model_parameters import BasicNet1, RPS_net_cifar, RPS_net_mlp, RPS_net
from utils.model_parameters import XDer, mammoth_load_checkpoint, get_backbone_class, get_dataset_class, original_cwd

device = "cuda:0" if torch.cuda.is_available() else "cpu"



def compute_accuracy(predictions, targets):
    correct, total = 0, 0
    correct += (predictions.cpu() == targets).sum()
    total += len(targets)
    correct = correct.cpu().data.numpy()
    return np.around(correct * 100 / total, decimals=2)


def generate_predictions(algorithm, model, ses, images, labels=None, **kwargs):

    images = images.to(device)

    if algorithm == "iTAML":
        model.set_shap(True)
        outputs2 = model(images)
        pred = torch.argmax(outputs2[:, 0:kwargs['cls_per_task'] * (1 + ses)], 1, keepdim=False)
    elif algorithm == "RPSnet":
        outputs = model(images, kwargs['infer_path'], -1)
        _, pred = torch.max(outputs, 1)
    elif algorithm == "DGR":
        real_scores = model.forward(images)
        _, pred = real_scores[:, 0: kwargs['cls_per_task'] * (1 + ses)].max(1)
    elif algorithm in pycil_algs:
        with torch.no_grad():
            outputs = model(images)["logits"]
        pred = torch.max(outputs, dim=1)[1]
    elif algorithm == "xder":
        outputs = model(images)
        _, pred = torch.max(outputs[:, :kwargs['cls_per_task']* (1 + ses)].data, 1)
    predicted = pred.squeeze()
    if labels is not None:
        acc = compute_accuracy(predicted, labels)
        print("Acc:", acc)

    return predicted

def load_model(algorithm, dataset, ses, shapArgs):
    model = None
    scholar = None
    model_path = ""
    alg_args = get_algorithm_args(algorithm, dataset)

    if algorithm in pycil_algs:
        model_path = f"saved_models/{algorithm}/{dataset}/{algorithm}_ses_{ses}.pth"
        load_start_sess = ses - 1 if algorithm == "foster" else 0
        match algorithm:
            case "foster":
                model = FOSTERNet(alg_args, False)
            case "memo":
                model = AdaptiveNet(alg_args, False)
            case "der":
                model = DERNet(alg_args, False)
            case "icarl":
                model = IncrementalNet(alg_args, False)
            case "ds-al":
                model = DSALNet(
                    alg_args,
                    alg_args["configurations"][f"{dataset}"]["buffer_size"],
                    alg_args["configurations"][f"{dataset}"]["gamma"],
                    alg_args["configurations"][f"{dataset}"]["gamma_comp"],
                    alg_args["configurations"][f"{dataset}"]["compensation_ratio"],
                )
                model.generate_buffer()
                model.generate_fc()
            case "tagfex":
                model = TagFexNet(alg_args, False)

        ###---Changes made to allow for non-uniform task class sizes and---###
        ###---correct dataset parameter------------------------------------###
        # Update the model architecture to match the task
        if ses > 0:
            model.update_fc(shapArgs.dataset_params.init_cls * (load_start_sess + 1))
            for i in range(load_start_sess+1, ses + 1):
                # I forgot to update to the dataset parameter before
                model.update_fc(shapArgs.dataset_params.class_per_task * (i + 1))
        else:
            model.update_fc(shapArgs.dataset_params.init_cls * (ses + 1))
        ###----------------------------------------------------------------###
    else:
        match algorithm:
            case "iTAML":
                model_path = f"saved_models/{algorithm}/{dataset}/session_{ses}_model_best.pth.tar"
                model = BasicNet1(alg_args, 0, device=device)
            case "RPSnet":
                model_path = f"saved_models/{algorithm}/{dataset}/session_{ses}_0_model_best.pth.tar"
                if dataset == "mnist":
                    model = RPS_net_mlp(alg_args)
                elif dataset == "imagenet200":
                    model = RPS_net(alg_args)
                else:
                    model = RPS_net_cifar(alg_args)
            case "xder":
                model_path = f"./{dataset}/xder_seq-{dataset}_ses_{ses}.pt"
                #model_path = f"./cifar10/xder_ses_{ses}.pt"
                #model_path = f"Saliency/xder/{dataset}/xder_ses_{ses}.pt"

                alg_args['dataset'] = f'seq-{dataset}'
                alg_args['num_classes'] = shapArgs.dataset_params.num_class
                args = argparse.Namespace(**alg_args)

                xder_dataset = get_dataset_class(args)
                backbone_cl, backbone_args = get_backbone_class('resnet18', return_args=True)
                parsed_args = {arg: getattr(args, arg) for arg in backbone_args.keys()}
                model = XDer(backbone_cl(**parsed_args),
                             xder_dataset.get_loss(),
                             args,
                             xder_dataset.get_transform(),
                             dataset=xder_dataset)

                model, _ = mammoth_load_checkpoint(model_path, model)
                model.eval()
                os.chdir(original_cwd)
                return model


    model_data = torch.load(model_path, map_location=device, weights_only=False)

    #print(model_data.keys())
    #print(model_data['state_dict'].keys())

    #print(model.state_dict().keys())
    if algorithm == "RPSnet":
        model.load_state_dict(model_data['state_dict'])
        model.eval()
    elif algorithm == "DGR" and scholar is not None:
        scholar.load_state_dict(model_data['state'])
        model = scholar.solver
    else:
        model.load_state_dict(model_data)
        model.eval()

    return model


#--------------------For iTAML--------------------#
args = iTAMLArgs
use_cuda = True if torch.cuda.is_available() else 'cpu'

def meta_test(model, memory, inc_dataset, testloader):
    all_models = []
    # switch to evaluate mode
    model.eval()

    meta_models = []
    base_model = copy.deepcopy(model)
    class_acc = {}
    meta_task_test_list = {}
    for task_idx in range(args.sess + 1):

        memory_data, memory_target = memory
        memory_data = np.array(memory_data, dtype="int32")
        memory_target = np.array(memory_target, dtype="int32")

        mem_idx = np.where((memory_target >= task_idx * args.class_per_task) & (
                    memory_target < (task_idx + 1) * args.class_per_task))[0]
        meta_memory_data = memory_data[mem_idx]
        meta_memory_target = memory_target[mem_idx]
        meta_model = copy.deepcopy(base_model)

        meta_loader = inc_dataset.get_custom_loader_idx(meta_memory_data, mode="train", batch_size=64)

        meta_optimizer = optim.Adam(meta_model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0,
                                    amsgrad=False)

        meta_model.train()

        # The range of classes that could be predicted in that task
        ai = args.class_per_task * task_idx
        bi = args.class_per_task * (task_idx + 1)
        # Total number of classes learned
        bb = args.class_per_task * (args.sess + 1)
        print("Training meta tasks:\t", task_idx)

        # META training
        if (args.sess != 0):
            for ep in range(1):
                for batch_idx, (inputs, targets) in enumerate(meta_loader):
                    targets_one_hot = torch.FloatTensor(inputs.shape[0], (task_idx + 1) * args.class_per_task)
                    targets_one_hot.zero_()
                    targets_one_hot.scatter_(1, targets[:, None], 1)
                    target_set = np.unique(targets)

                    if use_cuda:
                        inputs, targets_one_hot, targets = inputs.cuda(), targets_one_hot.cuda(), targets.cuda()
                    inputs, targets_one_hot, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(
                        targets_one_hot), torch.autograd.Variable(targets)

                    _, outputs = meta_model(inputs)
                    class_pre_ce = outputs.clone()
                    class_pre_ce = class_pre_ce[:, ai:bi]
                    class_tar_ce = targets_one_hot.clone()

                    loss = F.binary_cross_entropy_with_logits(class_pre_ce, class_tar_ce[:, ai:bi])

                    meta_optimizer.zero_grad()
                    loss.backward()
                    meta_optimizer.step()

        # META testing with given knowledge on task
        meta_model.eval()
        for cl in range(args.class_per_task):
            class_idx = cl + args.class_per_task * task_idx
            loader = inc_dataset.get_custom_loader_class([class_idx], mode="test", batch_size=10)

            for batch_idx, (inputs, targets) in enumerate(loader):
                targets_task = targets - args.class_per_task * task_idx

                if use_cuda:
                    inputs, targets_task = inputs.cuda(), targets_task.cuda()
                inputs, targets_task = torch.autograd.Variable(inputs), torch.autograd.Variable(targets_task)

                _, outputs = meta_model(inputs)

                if use_cuda:
                    inputs, targets = inputs.cuda(), targets_task.cuda()
                inputs, targets_task = torch.autograd.Variable(inputs), torch.autograd.Variable(targets_task)

                pred = torch.argmax(outputs[:, ai:bi], 1, keepdim=False)
                pred = pred.view(1, -1)
                correct = pred.eq(targets_task.view(1, -1).expand_as(pred)).view(-1)

                correct_k = float(torch.sum(correct).detach().cpu().numpy())

                for i, p in enumerate(pred.view(-1)):
                    key = int(p.detach().cpu().numpy())
                    key = key + args.class_per_task * task_idx
                    if (correct[i] == 1):
                        if (key in class_acc.keys()):
                            class_acc[key] += 1
                        else:
                            class_acc[key] = 1

        #           META testing - no knowledge on task
        meta_model.eval()
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            _, outputs = meta_model(inputs)
            outputs_base, _ = model(inputs)
            task_ids = outputs

            task_ids = task_ids.detach().cpu()
            outputs = outputs.detach().cpu()
            outputs = outputs.detach().cpu()
            outputs_base = outputs_base.detach().cpu()

            bs = inputs.size()[0]
            for i, t in enumerate(list(range(bs))):
                j = batch_idx * args.test_batch + i
                output_base_max = []
                for si in range(args.sess + 1):
                    sj = outputs_base[i][si * args.class_per_task:(si + 1) * args.class_per_task]
                    sq = torch.max(sj)
                    output_base_max.append(sq)

                task_argmax = np.argsort(outputs[i][ai:bi])[-5:]
                task_max = outputs[i][ai:bi][task_argmax]

                if (j not in meta_task_test_list.keys()):
                    meta_task_test_list[j] = [[task_argmax, task_max, output_base_max, targets[i]]]
                else:
                    meta_task_test_list[j].append([task_argmax, task_max, output_base_max, targets[i]])
        if args.sess == args.num_task-1:
            all_models.append(meta_model.to('cpu'))
            # Save the meta_model for each task_idx
            #torch.save(meta_model.state_dict(), f"meta_model_task_{task_idx}_session_{args.sess}.pth")
            #print(f"Saved adapted meta model for Task {task_idx} to {save_path}")
        elif args.sess != args.num_task-1 and task_idx == args.sess:
            all_models.append(meta_model.to('cpu'))
            # Save the adapted model for classes 4 and 5
            #torch.save(meta_model.state_dict(), 'meta_model_task2_classes4_5.pth')
        del meta_model

    '''
    acc_task = {}
    for i in range(args.sess + 1):
        acc_task[i] = 0
        for j in range(args.class_per_task):
            try:
                acc_task[i] += class_acc[i * args.class_per_task + j] / args.sample_per_task_testing[i] * 100
            except:
                pass
    print("\n".join([str(acc_task[k]).format(".4f") for k in acc_task.keys()]))
    print(class_acc)
    '''

    #print("Meta models:")
    #print(all_models)

    #return acc_task
    return all_models

def load_meta_models(dataset, sess):

    args = iTAMLArgs
    if dataset == "cifar100":
        args.class_per_task = 10
        args.num_class = 100
        args.num_task = 10
    else:
        args.class_per_task = 2
        args.num_class = 10
        args.num_task = 5

    args.dataset = dataset
    args.sess = sess
    args.data_path = f"Datasets/{dataset}/"
    inc_dataset = data.IncrementalDataset(
        dataset_name=args.dataset,
        args=args,
        random_order=args.random_classes,
        shuffle=True,
        seed=1,
        batch_size=args.train_batch,
        workers=0,#args.workers,
        validation_split=args.validation,
        increment=args.class_per_task,
    )

    #if (start_sess == ses and start_sess != 0):
    if args.sess != 0:
        inc_dataset._current_task = args.sess
        #with open(f"Saliency/iTAML/{dataset}" + "/sample_per_task_testing_" + str(args.sess - 1) + ".pickle", 'rb') as handle:
        #    sample_per_task_testing = pickle.load(handle)
        #inc_dataset.sample_per_task_testing = sample_per_task_testing
        #args.sample_per_task_testing = sample_per_task_testing

    memory = None
    if args.sess > 0:
        with open(f"saved_models/iTAML/{dataset}" + "/memory_" + str(args.sess - 1) + ".pickle", 'rb') as handle:
            memory = pickle.load(handle)

    _, _, _, testloader, for_memory = inc_dataset.new_task(memory)
    memory = inc_dataset.get_memory(memory, for_memory)
    model = load_model("iTAML", dataset, args.sess, args)
    return meta_test(model, memory, inc_dataset, testloader)
