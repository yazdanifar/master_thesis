import copy

from digits import network, loss
import torch
import torch.nn as nn


def cal_acc(loader, netF, netB, netC):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
    return accuracy * 100, mean_ent


def load_all_from_states(netF_state_list, netB_state_list, netC_state_list, args):
    netF_list, netB_list, netC_list = [], [], []
    for F_state, B_state, C_state in zip(netF_state_list, netB_state_list, netC_state_list):
        netF = network.DTNBase().cuda()

        netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features,
                                       bottleneck_dim=args.bottleneck).cuda()
        netC = network.feat_classifier(type=args.layer, class_num=args.class_num,
                                       bottleneck_dim=args.bottleneck).cuda()

        netF.load_state_dict(copy.deepcopy(F_state))
        netB.load_state_dict(copy.deepcopy(B_state))
        netC.load_state_dict(copy.deepcopy(C_state))

        netF_list.append(netF)
        netB_list.append(netB)
        netC_list.append(netC)

    return netF_list, netB_list, netC_list


def task_incremental_eval(test_loaders, netF_state_list, netB_state_list, netC_state_list, args):
    netF_list, netB_list, netC_list = load_all_from_states(netF_state_list, netB_state_list, netC_state_list, args)
    for net_list in [netF_list, netB_list, netC_list]:
        for net in net_list:
            net.eval()
    till_task = len(netF_state_list)
    accs = []
    for task in range(till_task):
        name = args.task_names[task]
        acc, _ = cal_acc(test_loaders[name], netF_list[task], netB_list[task], netC_list[task])
        accs.append(acc)
    return accs


def task_infer_predictor(inputs, netF_list, netB_list, netC_list):
    entropies = []
    outputs = []
    for netF, netB, netC in zip(netF_list, netB_list, netC_list):
        output = netC(netB(netF(inputs)))
        entropy = loss.Entropy(nn.Softmax(dim=1)(output)).unsqueeze(-1)
        entropies.append(entropy)
        outputs.append(output.unsqueeze(-1))
    preds = torch.cat(outputs, -1)
    inf_task_id = torch.cat(entropies, -1).argmin(-1)
    inf_task_id = inf_task_id.unsqueeze(-1).expand(-1, preds.size(1)).unsqueeze(-1)
    preds = torch.gather(preds, -1, inf_task_id).squeeze(-1)
    return preds


def task_agnostic_eval(test_loaders, netF_state_list, netB_state_list, netC_state_list, args):
    netF_list, netB_list, netC_list = load_all_from_states(netF_state_list, netB_state_list, netC_state_list, args)
    for net_list in [netF_list, netB_list, netC_list]:
        for net in net_list:
            net.eval()
    till_task = len(netF_state_list)
    accs = []
    for task in range(till_task):
        task_name = args.task_names[task]
        loader = test_loaders[task_name]
        start_test = True
        with torch.no_grad():
            iter_test = iter(loader)
            for i in range(len(loader)):
                data = iter_test.next()
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                preds = task_infer_predictor(inputs, netF_list, netB_list, netC_list)
                if start_test:
                    all_output = preds.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, preds.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
        _, predict = torch.max(all_output, 1)
        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
        accs.append(accuracy * 100)
    return accs
