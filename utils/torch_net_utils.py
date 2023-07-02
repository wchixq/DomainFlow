import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


def load_valid_model_zoo(model, pretrained_dict, skip=None):
    model_dict = model.state_dict()
    act_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and k not in skip}
    model_dict.update(act_dict)
    model.load_state_dict(model_dict)
    # print('pre-trained loaded')


def load_resnet_imagenet_pre_trained(net, pretrained):
    pretrained_dict = torch.load(pretrained)
    model_dict = net.state_dict()
    skip = ['fc.weight', 'fc.bias']
    act_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and k not in skip}
    model_dict.update(act_dict)
    net.load_state_dict(model_dict)


def save_state_dicts(net, save_name):
    torch.save(net.cpu().state_dict(), save_name)


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# adopted from LNFMM code ----------------------------------------------------------
def adjust_lr_epoch(optimizer, lr, cur_ep, epoch):
    """Sets the learning rate to the initial LR decayed by 0.5 every 30 epochs"""
    lr = lr * (0.5 ** (cur_ep // epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
# adopted from LNFMM code ----------------------------------------------------------


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
       lr += [param_group['lr']]
    return lr


def freeze_resnet_until_layer(net, until_layer):
    for name, child in net.named_children():
        if name is until_layer:
            break
        for param in child.parameters():
            param.requires_grad = False


def freeze_layer_param(net, layers):
    for name, child in net.named_children():
        if name in layers:
            for param in child.parameters():
                param.requires_grad = False


def set_train_mode_freeze_bn(net):
    net.train()
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def copy_all_params(net_target, net_source):
    for param_t, param_s in zip(net_target.parameters(), net_source.parameters()):
        param_t.data = param_s.data
    # BN not included in parameters
    for buf_t, buf_s in zip(net_target.buffers(), net_source.buffers()):
        buf_t.data = buf_s.data


def throw_net_grad(net):
    for param in net.parameters():
        param.requires_grad = False


def set_sep_optim_rate_for_layer(net, base_rate, spec_rate, layer_name=['fc',],
                                 optim_name='sgd', decay=0.0005, momentum=0.9):
    base_param = list()
    layer_param = list()
    for name, child in net.named_children():
        if name in layer_name:
            for param in child.parameters():
                layer_param.append(param)
        else:
            for param in child.parameters():
                base_param.append(param)
    if optim_name == 'sgd':
        optimizer = optim.SGD([{'params': base_param},
                               {'params': layer_param, 'lr': spec_rate}],
                              lr=base_rate, weight_decay=decay, momentum=momentum)
    elif optim_name == 'adam':
        optimizer = optim.Adam([{'params': base_param},
                               {'params': layer_param, 'lr': spec_rate}],
                               lr=base_rate, eps=1e-8, weight_decay=decay)
    else:
        print('Choose optimizer from SGD or Adam (invalid %s)' % optim_name)
        return
    return optimizer


def compact_sgdoptim_separate_params(param_list, rate_list, decay=0.0005, momentum=0.9):
    assert(len(param_list) == len(rate_list))
    param_dict = [{'params': param_list[0]}, ]
    for i in range(1, len(param_list)):
        param_dict.append({'params': param_list[i], 'lr': rate_list[i]})
    optimizer = optim.SGD(param_dict, lr=rate_list[0], weight_decay=decay, momentum=momentum)
    return optimizer


def find_spec_layer_params(net, layer_name=['fc',]):
    base_param = list()
    layer_param = list()
    for name, child in net.named_children():
        if name in layer_name:
            for param in child.parameters():
                layer_param.append(param)
        else:
            for param in child.parameters():
                base_param.append(param)
    return layer_param, base_param


# adapted from SAFN code ---------------------------------------------------------
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)
# adapted from SAFN code ---------------------------------------------------------


if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))
