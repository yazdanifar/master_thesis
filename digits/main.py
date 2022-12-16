import os, sys

from hypnettorch.hnets import ChunkedHMLP, HMLP, StructuredHMLP
from hypnettorch.utils import hnet_regularizer as hreg

from digits.state_tensor_translator import StateTensorTranslator

sys.path.append('..')
import argparse
import os.path as osp
from collections import OrderedDict
import functools

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import network, loss
import random, copy
from scipy.spatial.distance import cdist
from digits.data_preparation import digit_load
from digits.evaluation import cal_acc, task_incremental_eval, task_agnostic_eval


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def meta_lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
    return optimizer


def train_source(train_loader, test_loader, args, task_label):
    ## set base network
    netF = network.DTNBase().cuda()

    netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    param_group = []
    learning_rate = args.lr
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    acc_init = 0
    max_iter = args.max_epoch * len(train_loader)
    interval_iter = max_iter // 10
    iter_num = 0

    netF.train()
    netB.train()
    netC.train()

    while iter_num < max_iter:
        try:
            inputs_source, labels_source, _ = iter_source.next()
        except:
            iter_source = iter(train_loader)
            inputs_source, labels_source, _ = iter_source.next()

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
        outputs_source = netC(netB(netF(inputs_source)))
        classifier_loss = loss.CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth) \
            (outputs_source, labels_source)
        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()
        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            acc_s_te, _ = cal_acc(test_loader, netF, netB, netC)
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(task_label, iter_num, max_iter,
                                                                        acc_s_te)
            args.out_file.write(log_str + '\n')
            print(log_str)

            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                best_netF = copy.deepcopy(netF.state_dict())
                best_netB = copy.deepcopy(netB.state_dict())
                best_netC = copy.deepcopy(netC.state_dict())

            netF.train()
            netB.train()
            netC.train()

    netF.load_state_dict(best_netF)
    netB.load_state_dict(best_netB)
    netC.load_state_dict(best_netC)

    return netF, netB, netC


def train_single_src(shuffled_loader, un_shuffled_loader, test_loader,
                     netF_state, netB_state, netC_state, args, task_label):
    ## set base network
    netF = network.DTNBase().cuda()

    netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    netF.load_state_dict(copy.deepcopy(netF_state))
    netB.load_state_dict(copy.deepcopy(netB_state))
    netC.load_state_dict(copy.deepcopy(netC_state))

    netC.eval()
    for k, v in netC.named_parameters():
        v.requires_grad = False

    param_group = []
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(shuffled_loader)
    interval_iter = len(shuffled_loader)
    # interval_iter = max_iter // args.interval
    iter_num = 0
    best_acc = 0
    while iter_num < max_iter:
        optimizer.zero_grad()
        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(shuffled_loader)
            inputs_test, _, tar_idx = iter_test.next()

        if inputs_test.size(0) == 1:
            continue

        if iter_num % interval_iter == 0 and args.cls_par > 0:
            netF.eval()
            netB.eval()
            mem_label = single_obtain_label(un_shuffled_loader, netF, netB, netC, args)
            mem_label = torch.from_numpy(mem_label).cuda()
            netF.train()
            netB.train()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_test = inputs_test.cuda()
        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)

        if args.cls_par > 0:
            pred = mem_label[tar_idx]
            classifier_loss = args.cls_par * nn.CrossEntropyLoss()(outputs_test, pred)
        else:
            classifier_loss = torch.tensor(0.0).cuda()

        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                entropy_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))

            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            acc, _ = cal_acc(test_loader, netF, netB, netC)
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(task_label, iter_num, max_iter, acc)
            args.out_file.write(log_str + '\n')
            print(log_str)
            if acc >= best_acc:
                best_acc = acc
                best_F = copy.deepcopy(netF.state_dict())
                best_B = copy.deepcopy(netB.state_dict())
                best_C = copy.deepcopy(netC.state_dict())
            netF.train()
            netB.train()
            netF.load_state_dict(best_F)
            netB.load_state_dict(best_B)
            netC.load_state_dict(best_C)

    log_str = 'task {} best acc: {:.2f}'.format(task_label, best_acc)
    args.out_file.write(log_str + '\n')
    print(log_str)
    return netF, netB, netC


def single_obtain_label(loader, netF, netB, netC, args, c=None):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    all_fea = all_fea.float().cpu().numpy()

    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
    dd = cdist(all_fea, initc, 'cosine')
    pred_label = dd.argmin(axis=1)
    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)
        acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
    args.out_file.write(log_str + '\n')
    print(log_str)
    return pred_label.astype('int')


def train_multi_src(shuffeled_loader, un_shuffled_loader, test_loader,
                    netF_state_list, netB_state_list, netC_state_list, task_label, args):
    ## set base network
    netF_list = [network.DTNBase().cuda() for _ in range(len(netF_state_list))]

    w = 2 * torch.rand((len(netF_state_list),)) - 1

    netB_list = [network.feat_bottleneck(type=args.classifier, feature_dim=netF_list[i].in_features,
                                         bottleneck_dim=args.bottleneck).cuda() for i in range(len(netF_state_list))]
    netC_list = [
        network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda() for _
        in range(len(netF_state_list))]
    netG_list = [network.scalar(w[i]).cuda() for i in range(len(netF_state_list))]

    param_group = []
    for i in range(len(netF_state_list)):
        netF_list[i].load_state_dict(copy.deepcopy(netF_state_list[i]))
        netF_list[i].eval()
        for k, v in netF_list[i].named_parameters():
            param_group += [{'params': v, 'lr': args.lr}]
        netB_list[i].load_state_dict(copy.deepcopy(netB_state_list[i]))
        netB_list[i].eval()
        for k, v in netB_list[i].named_parameters():
            param_group += [{'params': v, 'lr': args.lr}]

        netC_list[i].load_state_dict(copy.deepcopy(netC_state_list[i]))
        netC_list[i].eval()
        for k, v in netC_list[i].named_parameters():
            v.requires_grad = False

        for k, v in netG_list[i].named_parameters():
            param_group += [{'params': v, 'lr': args.alpha_lr}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(shuffeled_loader)
    interval_iter = len(shuffeled_loader)
    # interval_iter = max_iter // args.interval
    iter_num = 0
    best_acc = 0
    while iter_num < max_iter:
        optimizer.zero_grad()
        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(shuffeled_loader)
            inputs_test, _, tar_idx = iter_test.next()

        if inputs_test.size(0) == 1:
            continue

        if iter_num % interval_iter == 0 and args.cls_par > 0:
            initc = []
            all_feas = []
            for i in range(len(netF_state_list)):
                netF_list[i].eval()
                netB_list[i].eval()
                temp1, temp2 = obtain_label(un_shuffled_loader, netF_list[i], netB_list[i], netC_list[i], args)
                temp1 = torch.from_numpy(temp1).cuda()
                temp2 = torch.from_numpy(temp2).cuda()
                initc.append(temp1)
                all_feas.append(temp2)
                netF_list[i].train()
                netB_list[i].train()

        inputs_test = inputs_test.cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        outputs_all = torch.zeros(len(netF_state_list), inputs_test.shape[0], args.class_num)
        weights_all = torch.ones(inputs_test.shape[0], len(netF_state_list))
        outputs_all_w = torch.zeros(inputs_test.shape[0], args.class_num)
        init_ent = torch.zeros(1, len(netF_state_list))

        for i in range(len(netF_state_list)):
            features_test = netB_list[i](netF_list[i](inputs_test))
            outputs_test = netC_list[i](features_test)
            softmax_ = nn.Softmax(dim=1)(outputs_test)
            ent_loss = torch.mean(loss.Entropy(softmax_))
            init_ent[:, i] = ent_loss
            weights_test = netG_list[i](features_test)
            outputs_all[i] = outputs_test
            weights_all[:, i] = weights_test.squeeze()

        z = torch.sum(weights_all, dim=1)
        z = z + 1e-16

        weights_all = torch.transpose(torch.transpose(weights_all, 0, 1) / z, 0, 1)
        outputs_all = torch.transpose(outputs_all, 0, 1)

        z_ = torch.sum(weights_all, dim=0)

        z_2 = torch.sum(weights_all)
        z_ = z_ / z_2

        for i in range(inputs_test.shape[0]):
            outputs_all_w[i] = torch.matmul(torch.transpose(outputs_all[i], 0, 1), weights_all[i])

        if args.cls_par > 0:
            initc_ = torch.zeros(initc[0].size()).cuda()
            temp = all_feas[0]
            all_feas_ = torch.zeros(temp[tar_idx, :].size()).cuda()
            for i in range(len(netF_state_list)):
                initc_ = initc_ + z_[i] * initc[i].float()
                src_fea = all_feas[i]
                all_feas_ = all_feas_ + z_[i] * src_fea[tar_idx, :]
            dd = torch.cdist(all_feas_.float(), initc_.float(), p=2)
            pred_label = dd.argmin(dim=1)
            pred_label = pred_label.int()
            pred = pred_label.long()
            classifier_loss = args.cls_par * nn.CrossEntropyLoss()(outputs_all_w, pred.cpu())
        else:
            classifier_loss = torch.tensor(0.0)

        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_all_w)
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                entropy_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))

            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            for i in range(len(netF_state_list)):
                netF_list[i].eval()
                netB_list[i].eval()
            acc, _ = cal_acc_multi(test_loader, netF_list, netB_list, netC_list, netG_list, args)
            log_str = 'Task {} Adaptation, Iter:{}/{}; Accuracy = {:.2f}%'.format(task_label, iter_num, max_iter, acc)
            args.out_file.write(log_str + '\n')
            print(log_str)
            if acc >= best_acc:
                best_acc = acc
                best_F_states = [copy.deepcopy(net.state_dict()) for net in netF_list]
                best_B_states = [copy.deepcopy(net.state_dict()) for net in netB_list]
                best_C_states = [copy.deepcopy(net.state_dict()) for net in netC_list]
                best_G_states = [copy.deepcopy(net.state_dict()) for net in netG_list]
    for net, state in zip(netF_list, best_F_states):
        net.load_state_dict(state)
    for net, state in zip(netB_list, best_B_states):
        net.load_state_dict(state)
    for net, state in zip(netC_list, best_C_states):
        net.load_state_dict(state)
    for net, state in zip(netG_list, best_G_states):
        net.load_state_dict(state)
    log_str = 'task {} adaptation best acc: {:.2f}'.format(task_label, best_acc)
    args.out_file.write(log_str + '\n')
    print(log_str)
    netF, netB, netC = distill(netF_list, netB_list, netC_list, netG_list, shuffeled_loader, test_loader, args,
                               task_label)

    return netF, netB, netC


def distill(netF_list, netB_list, netC_list, netG_list, dset_loader, test_loader, args, task_label):
    all_net_lists = netF_list, netB_list, netC_list, netG_list
    for net_list in all_net_lists:
        for i in range(len(netF_list)):
            net_list[i].eval()
            for k, v in net_list[i].named_parameters():
                v.requires_grad = False
    netF, netB, netC = student_factory(args.stu_init, netF_list, netB_list, netC_list, netG_list)

    param_group = []
    learning_rate = args.distill_lr
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    acc_init = 0
    max_iter = args.max_epoch * len(dset_loader)
    interval_iter = max_iter // 10
    iter_num = 0

    netF.train()
    netB.train()
    netC.train()

    while iter_num < max_iter:
        try:
            inputs = iter_source.next()
        except:
            iter_source = iter(dset_loader)
            inputs = iter_source.next()

        inputs = inputs[0]
        if inputs.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        labels, logits = aggregate_preds(inputs, netF_list, netB_list, netC_list, netG_list)

        inputs, labels, logits = inputs.cuda(), labels.cuda(), logits.cuda()
        labels, logits = labels.detach(), logits.detach()
        outputs = netC(netB(netF(inputs)))
        classifier_loss = nn.CrossEntropyLoss()(outputs, labels)

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            acc_s_te, _ = cal_acc(test_loader, netF, netB, netC)
            log_str = 'Task {} Distillation, Iter:{}/{}; Accuracy = {:.2f}%'.format(task_label, iter_num, max_iter,
                                                                                    acc_s_te)
            args.out_file.write(log_str + '\n')
            print(log_str)

            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                best_netF = copy.deepcopy(netF.state_dict())
                best_netB = copy.deepcopy(netB.state_dict())
                best_netC = copy.deepcopy(netC.state_dict())

            netF.train()
            netB.train()
            netC.train()
    log_str = 'task {} distillation best acc: {:.2f}'.format(task_label, acc_init)
    args.out_file.write(log_str + '\n')
    print(log_str)
    netF.load_state_dict(best_netF)
    netB.load_state_dict(best_netB)
    netC.load_state_dict(best_netC)

    return netF, netB, netC


def student_factory(init_strategy, netF_list, netB_list, netC_list, netG_list):
    assert init_strategy in ['scratch', 'most_rel', 'alpha_comb']
    netF = network.DTNBase().cuda()
    netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num,
                                   bottleneck_dim=args.bottleneck).cuda()
    with torch.no_grad():
        alphas = [g(torch.ones([1, 1], device='cuda')) for g in netG_list]
    alphas = [alpha / sum(alphas) for alpha in alphas]
    if init_strategy == 'most_rel':
        best_index = torch.cat(alphas).argmax(0)
        netF_state, netB_state, netC_state = netF_list[best_index].state_dict(), \
                                             netB_list[best_index].state_dict(), netC_list[best_index].state_dict()
        netF.load_state_dict(netF_state)
        netB.load_state_dict(netB_state)
        netC.load_state_dict(netC_state)
    if init_strategy == 'alpha_comb':
        netF_state = {k: sum([alphas[i].squeeze() * netF_list[i].state_dict().get(k) for i in range(len(netF_list))])
                      for k in netF_list[0].state_dict().keys()}
        netB_state = {k: sum([alphas[i].squeeze() * netB_list[i].state_dict().get(k) for i in range(len(netB_list))])
                      for k in netB_list[0].state_dict().keys()}
        netC_state = {k: sum([alphas[i].squeeze() * netC_list[i].state_dict().get(k) for i in range(len(netC_list))])
                      for k in netC_list[0].state_dict().keys()}
        netF.load_state_dict(OrderedDict(netF_state))
        netB.load_state_dict(OrderedDict(netB_state))
        netC.load_state_dict(OrderedDict(netC_state))

    return netF, netB, netC


def aggregate_preds(inputs, netF_list, netB_list, netC_list, netG_list):
    with torch.no_grad():
        inputs = inputs.cuda()
        outputs_all = torch.zeros(len(netF_list), inputs.shape[0], args.class_num)
        weights_all = torch.ones(inputs.shape[0], len(netF_list))
        outputs_all_w = torch.zeros(inputs.shape[0], args.class_num)

        for i in range(len(netF_list)):
            features = netB_list[i](netF_list[i](inputs))
            outputs = netC_list[i](features)
            weights = netG_list[i](features)
            outputs_all[i] = outputs
            weights_all[:, i] = weights.squeeze()

        z = torch.sum(weights_all, dim=1)
        z = z + 1e-16

        weights_all = torch.transpose(torch.transpose(weights_all, 0, 1) / z, 0, 1)
        # print(weights_all.mean(dim=0))
        outputs_all = torch.transpose(outputs_all, 0, 1)
        for i in range(inputs.shape[0]):
            outputs_all_w[i] = torch.matmul(torch.transpose(outputs_all[i], 0, 1), weights_all[i])

        all_output = outputs_all_w.float().cpu()

    _, predict = torch.max(all_output, 1)

    return predict, all_output


def obtain_label(loader, netF, netB, netC, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs.float()))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    all_fea = all_fea.float().cpu().numpy()

    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])

    dd = cdist(all_fea, initc, 'cosine')
    pred_label = dd.argmin(axis=1)
    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)
        acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
    args.out_file.write(log_str + '\n')
    print(log_str)
    return initc, all_fea


def cal_acc_multi(loader, netF_list, netB_list, netC_list, netG_list, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs_all = torch.zeros(len(netF_list), inputs.shape[0], args.class_num)
            weights_all = torch.ones(inputs.shape[0], len(netF_list))
            outputs_all_w = torch.zeros(inputs.shape[0], args.class_num)

            for i in range(len(netF_list)):
                features = netB_list[i](netF_list[i](inputs))
                outputs = netC_list[i](features)
                weights = netG_list[i](features)
                outputs_all[i] = outputs
                weights_all[:, i] = weights.squeeze()

            z = torch.sum(weights_all, dim=1)
            z = z + 1e-16

            weights_all = torch.transpose(torch.transpose(weights_all, 0, 1) / z, 0, 1)
            outputs_all = torch.transpose(outputs_all, 0, 1)

            for i in range(inputs.shape[0]):
                outputs_all_w[i] = torch.matmul(torch.transpose(outputs_all[i], 0, 1), weights_all[i])

            if start_test:
                all_output = outputs_all_w.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs_all_w.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
    return accuracy * 100, mean_ent


def load_models(task_id, args):
    netF = network.DTNBase().cuda()

    netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    netF.load_state_dict(torch.load(osp.join(args.output, "{}_F.pt".format(task_id))))
    netB.load_state_dict(torch.load(osp.join(args.output, "{}_B.pt".format(task_id))))
    netC.load_state_dict(torch.load(osp.join(args.output, "{}_C.pt".format(task_id))))
    return netF, netB, netC


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    s += "\n=========================================="
    return s


def phi_loss(task_id, hnet, weights):
    W_target = torch.cat([w.view(-1) for w in weights])
    weights_predicted = hnet.forward(cond_id=task_id)
    W_predicted = torch.cat([w.view(-1) for w in weights_predicted])
    return (W_target - W_predicted).pow(2).sum()


def load_prev_task_weights(hnet, task_id):
    netF_states, netB_states, netC_states = [], [], []
    for t in range(task_id):
        tensors = hnet(cond_id=t)
        netF_state, netB_state, netC_state = translator.tensor_to_state(tensors)
        netF_states.append(netF_state)
        netB_states.append(netB_state)
        netC_states.append(netC_state)

    return netF_states, netB_states, netC_states


def meta_learn_nets(netF, netB, netC, hnet, task_id):

    meta_opt = optim.Adam(hnet.parameters(), lr=args.meta_lr[task_id], weight_decay=args.meta_wd[task_id])
    meta_opt = op_copy(meta_opt)
    phi = translator.states_to_tensor(netF.state_dict(), netB.state_dict(), netC.state_dict())
    phi = [p.cuda() for p in phi]
    prev_hnet_theta = [p.detach().clone() for p in hnet.unconditional_params]
    prev_task_embs = [p.detach().clone() for p in hnet.conditional_params]
    log_interval = 100
    eval_interval = 500
    for e in range(args.meta_epochs[task_id]):
        meta_lr_scheduler(meta_opt, iter_num=e,
                                       max_iter=args.meta_epochs[task_id],
                                       gamma=args.meta_gamma[task_id],
                                       power=args.meta_power[task_id])

        meta_opt.zero_grad()
        meta_loss = phi_loss(task_id, hnet, phi)
        if task_id > 0 and not args.no_replay:
            replay_loss = args.replay_coeff * hreg.calc_fix_target_reg(hnet, task_id, prev_theta=prev_hnet_theta,
                                                  prev_task_embs=prev_task_embs, inds_of_out_heads=None,
                                                  batch_size=-1)
            if e % log_interval == 0 or e == (args.meta_epochs[task_id] - 1):
                print(f"task: {task_id} epoch: {e:4d} meta loss: {meta_loss:,.0f} "
                      f"replay loss: {replay_loss:,.0f}")
            meta_loss += replay_loss
        else:
            if e % log_interval == 0 or e == (args.meta_epochs[task_id] - 1):
                print(f"task: {task_id} epoch: {e:4d} meta loss: {meta_loss:,.0f}")

        meta_loss.backward()
        meta_opt.step()

        if e % eval_interval == 0 or e == (args.meta_epochs[task_id] - 1):
            evaluate_over_all_performance(hnet, task_id, args)
    return phi


def evaluate_over_all_performance(hnet, task_id, args):
    mode = hnet.training
    hnet.eval()
    with torch.no_grad():
        netF_states, netB_states, netC_states = load_prev_task_weights(hnet, task_id + 1)
    hnet.train(mode)

    log_str = ''
    if args.eval_scenario in ['both', 'TI']:
        accs = task_incremental_eval(test_loaders, netF_states, netB_states, netC_states, args)
        log_str += 'task incremental results:'
        for t, acc in enumerate(accs):
            log_str += ' task {}: {:.2f}'.format(t, acc)
        log_str += '\n'

    if args.eval_scenario in ['both', 'DI']:
        accs = task_agnostic_eval(test_loaders, netF_states, netB_states, netC_states, args)
        log_str += 'domain incremental results:'
        for t, acc in enumerate(accs):
            log_str += ' task {}: {:.2f}'.format(t, acc)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--task_names', type=str, required=True, help="dataset names i.e: smu")
    parser.add_argument('--max_epoch', type=int, default=30, help="maximum epoch")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--alpha_lr', type=float, default=0.01, help="alpha learning rate")
    parser.add_argument('--distill_lr', type=float, default=0.01, help="distillation learning rate")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--cls_par', type=float, default=0.1)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--stu_init', type=str, default='alpha_comb', choices=["alpha_comb", "scratch", "most_rel"])
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--src_only', action='store_true',
                        help='if specified, only source will be used for all the tasks')




    parser.add_argument('--meta_epochs', type=int, nargs='+', default=[5000, 5000, 5000, 5000])
    parser.add_argument('--meta_lr', type=float, nargs='+', default=[0.1, 0.1, 0.1])
    parser.add_argument('--meta_gamma', type=int, nargs='+', default=[50, 80, 20])
    parser.add_argument('--meta_power', type=float, nargs='+', default=[.9, 1.5, 1.5])
    parser.add_argument('--replay_coeff', type=float, default=1)
    parser.add_argument('--meta_wd', type=float, nargs='+', default=[1, 1, 1], help="meta weight decay")

    parser.add_argument('--no_replay', action='store_true', help="don't meta replay")
    parser.add_argument('--eval_scenario', type=str, default="TI", choices=["both", "TI", "DI"])
    parser.add_argument('--output', type=str, default='ckps_digits')
    parser.add_argument('--exp_name', type=str)
    args = parser.parse_args()
    args.class_num = 10

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    shuffled_loaders, un_shuffled_loaders, test_loaders = digit_load(args)
    src_name = args.task_names[0]

    args.exp_name = args.task_names if args.exp_name is None else args.exp_name
    args.output = osp.join(args.output, args.exp_name, 'seed' + str(args.seed))

    if not osp.exists(args.output):
        os.system('mkdir -p ' + args.output)
    if not osp.exists(args.output):
        os.mkdir(args.output)
    args.out_file = open(osp.join(args.output, 'log.txt'), 'w')
    args.out_file.write(print_args(args) + '\n')
    args.out_file.flush()
    log_str = 'task 0 started'
    args.out_file.write(log_str + '\n')
    print(log_str)
    if not osp.exists(osp.join(args.output, '0_F.pt')):
        netF, netB, netC = train_source(shuffled_loaders[src_name], test_loaders[src_name], args, src_name)
        torch.save(netF.state_dict(), osp.join(args.output, "0_F.pt"))
        torch.save(netB.state_dict(), osp.join(args.output, "0_B.pt"))
        torch.save(netC.state_dict(), osp.join(args.output, "0_C.pt"))
    else:
        netF, netB, netC = load_models(0, args)

    translator = StateTensorTranslator(netF.state_dict(), netB.state_dict(), netC.state_dict())
    hnet = ChunkedHMLP(target_shapes=translator.tensor_shapes(), chunk_size=7000,
                       chunk_emb_size=100, cond_chunk_embs=True,
                       cond_in_size=0, layers=[], verbose=True, num_cond_embs=len(args.task_names)).cuda()

    nn.init.ones_(hnet._internal_params[0])
    nn.init.zeros_(hnet._internal_params[1])

    meta_learn_nets(netF, netB, netC, hnet, 0)

    log_str = 'task 1 started'
    args.out_file.write(log_str + '\n')
    print(log_str)

    task_name = args.task_names[1]
    if not osp.exists(osp.join(args.output, '1_F.pt')):
        netF, netB, netC = train_single_src(shuffled_loaders[task_name], un_shuffled_loaders[task_name],
                                            test_loaders[task_name], netF.state_dict(), netB.state_dict(),
                                            netC.state_dict(), args, task_name)
        torch.save(netF.state_dict(), osp.join(args.output, "1_F.pt"))
        torch.save(netB.state_dict(), osp.join(args.output, "1_B.pt"))
        torch.save(netC.state_dict(), osp.join(args.output, "1_C.pt"))
    else:
        netF, netB, netC = load_models(1, args)

    meta_learn_nets(netF, netB, netC, hnet, 1)
    for task_id in range(2, len(args.task_names)):
        hnet.eval()
        with torch.no_grad():
            netF_states, netB_states, netC_states = load_prev_task_weights(hnet, task_id)
        hnet.train()
        log_str = 'task {} started'.format(task_id)
        args.out_file.write(log_str + '\n')
        print(log_str)
        task_name = args.task_names[task_id]
        if not osp.exists(osp.join(args.output, '{}_F.pt'.format(task_id))):
            if args.src_only:
                netF, netB, netC = train_single_src(shuffled_loaders[task_name], un_shuffled_loaders[task_name],
                                                    test_loaders[task_name],
                                                    netF_states[0], netB_states[0], netC_states[0], args, task_name)
            else:
                netF, netB, netC = train_multi_src(shuffled_loaders[task_name], un_shuffled_loaders[task_name],
                                                   test_loaders[task_name], netF_states, netB_states, netC_states,
                                                   task_name, args)
            torch.save(netF.state_dict(), osp.join(args.output, "{}_F.pt".format(task_id)))
            torch.save(netB.state_dict(), osp.join(args.output, "{}_B.pt".format(task_id)))
            torch.save(netC.state_dict(), osp.join(args.output, "{}_C.pt".format(task_id)))
        else:
            netF, netB, netC = load_models(task_id, args)
        meta_learn_nets(netF, netB, netC, hnet, task_id)
