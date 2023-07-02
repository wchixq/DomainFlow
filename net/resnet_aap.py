import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ROOT_DIR
MODEL_DIR = ROOT_DIR + '/models'


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck_TN(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, TN_idx=('1','2')):
        super(Bottleneck_TN, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if TN_idx is not None and '1' in TN_idx:
            self.bn1 = nn.InstanceNorm1d(planes, affine=True)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        if TN_idx is not None and '2' in TN_idx:
            self.bn2 = nn.InstanceNorm1d(planes, affine=True)
        else:
            self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        if TN_idx is not None and '3' in TN_idx:
            self.bn3 = nn.InstanceNorm1d(planes * 4, affine=True)
        else:
            self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def apply_bn(self, bn, x):
        if isinstance(bn, nn.InstanceNorm1d):
            if not self.training:
                T_per_card = 1
            else:
                T_per_card = 2
            N_s = x.shape[0]
            N_f = x.shape[-1]
            x = x.view((T_per_card, x.shape[1],-1))
            x = bn(x)
            x = x.view((N_s,x.shape[1],N_f, N_f))
        else:
            x = bn(x)

        return x

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.apply_bn(self.bn1, out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.apply_bn(self.bn2, out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.apply_bn(self.bn3, out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BottleneckExp(nn.Module):

    def __init__(self, inplanes, planes, stride=1, expansion=4, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.expansion = expansion

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    inplanes = 64

    def __init__(self, block, num_blocks, num_classes=100, in_shape=(3, 224, 224), temperature=1.0):
        super().__init__()
        in_channels, height, width = in_shape

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1  = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64,  num_blocks[0])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.dropout = nn.Dropout(p=0.5)
        self.temperature = temperature

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward_fc(self, x):
        return self.fc(x)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #  added Dropout
        x = self.dropout(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.view(x.size(0), -1)
        # added Dropout
        x = self.dropout(x)
        x = self.fc(x)
        x = x / self.temperature
        return x

    def forward_out_flatfeat(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 64, 56, 56 --> 200704

        x = self.layer1(x)  # 256, 56, 56 --> 802816
        x = self.layer2(x)  # 512, 28, 28 --> 401408
        x = self.layer3(x)  # 1024, 14, 14 --> 200704
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.view(x.size(0), -1)
        return x

    def forward_out_feat(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 64, 56, 56 --> 200704

        x = self.layer1(x)  # 256, 56, 56 --> 802816
        x = self.layer2(x)  # 512, 28, 28 --> 401408
        x = self.layer3(x)  # 1024, 14, 14 --> 200704
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return x

    def forward_layer4_feats(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 64, 56, 56 --> 200704

        x = self.layer1(x)  # 256, 56, 56 --> 802816
        x = self.layer2(x)  # 512, 28, 28 --> 401408
        x = self.layer3(x)  # 1024, 14, 14 --> 200704
        x = self.layer4(x)
        return x

    def forward_with_flatfeat(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.view(x.size(0), -1)
        logit = self.fc(x)
        return logit, x

    def forward_with_feat(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        logit = x.view(x.size(0), -1)
        logit = self.fc(logit)
        return logit, x

    def forward_with_flat_l1feat(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x0 = self.layer2(x)
        x0 = self.layer3(x0)
        x0 = self.layer4(x0)
        x0 = F.adaptive_avg_pool2d(x0, output_size=1)
        x0 = x0.flatten(start_dim=1)
        x0 = self.fc(x0)
        x0 = x0 / self.temperature
        # flatten feature before return
        x = x.flatten(start_dim=1)
        return x0, x  # FC, layer1

    def forward_flat_l1feat(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        # flatten feature before return
        x = x.flatten(start_dim=1)
        return x

    # block dropout to estimate model uncertainty -----------------
    def forward_flat_l1feat_drop(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.layer1(x)
        x = self.dropout(x)
        # flatten feature before return
        x = x.flatten(start_dim=1)
        return x

    # multiple dropout on layer 1 feature only for feature net uncertainty
    def forward_with_flat_l1feat_drop(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.layer1(x)
        x = self.dropout(x)
        x0 = self.layer2(x)
        # x0 = self.dropout(x0)
        x0 = self.layer3(x0)
        # x0 = self.dropout(x0)
        x0 = self.layer4(x0)
        # x0 = self.dropout(x0)
        x0 = F.adaptive_avg_pool2d(x0, output_size=1)
        x0 = x0.flatten(start_dim=1)
        x0 = self.fc(x0)
        x0 = x0 / self.temperature
        # flatten layer 1 feature before return
        x = x.flatten(start_dim=1)
        return x0, x  # FC, layer1
    # block dropout to estimate model uncertainty -----------------


def load_resnet_imagenet_pre_trained(net, pretrained):
    net.load_state_dict(torch.load(pretrained), strict=False)

    pretrained_dict = torch.load(pretrained)
    model_dict = net.state_dict()
    skip = ['fc.weight', 'fc.bias']
    act_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and k not in skip}
    model_dict.update(act_dict)
    net.load_state_dict(model_dict)


def resnet18(**kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    pretrained = MODEL_DIR + '/resnet18-5c106cde.pth'
    model.load_state_dict(torch.load(pretrained), strict=False)
    # load_resnet_imagenet_pre_trained(model, MODEL_DIR + '/resnet18-5c106cde.pth')
    return model


def resnet34(**kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    pretrained = MODEL_DIR + '/resnet34-333f7ec4.pth'
    model.load_state_dict(torch.load(pretrained), strict=False)
    # load_resnet_imagenet_pre_trained(model, MODEL_DIR + '/resnet34-333f7ec4.pth')
    return model


def resnet50(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    pretrained = MODEL_DIR + '/resnet50-19c8e357.pth'
    model.load_state_dict(torch.load(pretrained), strict=False)
    # load_resnet_imagenet_pre_trained(model, MODEL_DIR + '/resnet50-19c8e357.pth')
    return model


def resnet101(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    pretrained = MODEL_DIR + '/resnet101-5d3b4d8f.pth'
    model.load_state_dict(torch.load(pretrained), strict=False)
    # load_resnet_imagenet_pre_trained(model, MODEL_DIR + '/resnet101-5d3b4d8f.pth')
    return model


def resnet152(**kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    pretrained = MODEL_DIR + '/resnet152-b121ed2d.pth'
    model.load_state_dict(torch.load(pretrained), strict=False)
    # load_resnet_imagenet_pre_trained(model, MODEL_DIR + '/resnet152-b121ed2d.pth')
    return model


class ResFeatExtractor(nn.Module):
    inplanes = 64

    def __init__(self, block, num_blocks, in_shape=(3, 224, 224)):
        super().__init__()
        in_channels, height, width = in_shape

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1  = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64,  num_blocks[0])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # 
        self.dropout = nn.Dropout(p=0.5)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward_train(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # added Dropout (should only use in training!!)
        # dropout: train and test mode
        x = self.dropout(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # N * 64, 56, 56 --> 200704

        x = self.layer1(x)  # N * 256, 56, 56 --> 802816
        x = self.layer2(x)  # N * 512, 28, 28 --> 401408
        x = self.layer3(x)  # N * 1024, 14, 14 --> 200704
        x = self.layer4(x)  # N * 2048, 7 , 7 --> 100352
        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.view(x.size(0), -1)  # N * 2048
        return x

    def forward_fromlayer1(self, x):
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.view(x.size(0), -1)
        return x

    def forward_withl1feat(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x1 = self.layer2(x)
        x1 = self.layer3(x1)
        x1 = self.layer4(x1)
        x1 = F.adaptive_avg_pool2d(x1, output_size=1)
        x1 = x1.view(x1.size(0), -1)
        return x1, x

    def forward_fromlayer3(self, x):
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.view(x.size(0), -1)
        return x

    def forward_withl1l3feat(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x1 = self.layer2(x)
        x1 = self.layer3(x1)
        x2 = self.layer4(x1)
        x2 = F.adaptive_avg_pool2d(x2, output_size=1)
        x2 = x2.view(x2.size(0), -1)
        return x2, x, x1

    def forward_withlp11feat(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x2 = self.layer3(x2)
        x2 = self.layer4(x2)
        x2 = F.adaptive_avg_pool2d(x2, output_size=1)
        x2 = x2.view(x2.size(0), -1)
        return x2, x, x1

    def forward_frompool1(self, x):
        x = self.layer1(x)  # N * 256, 56, 56 --> 802816
        x = self.layer2(x)  # N * 512, 28, 28 --> 401408
        x = self.layer3(x)  # N * 1024, 14, 14 --> 200704
        x = self.layer4(x)  # N * 2048, 7 , 7 --> 100352
        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.view(x.size(0), -1)  # N * 2048
        return x

    def forward_drop(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x = self.layer3(x)
        x = self.dropout(x)
        x = self.layer4(x)
        x = self.dropout(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.view(x.size(0), -1)
        return x

    def forward_to_layer2(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        return x

    def forward_to_pool1(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

    def forward_to_pool1_layer1(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        return x, x1

    def forward_to_layer4(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward_from_layer2(self, x):
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward_l1feat(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        return x


def resnet18backbone(**kwargs):
    model = ResFeatExtractor(BasicBlock, [2, 2, 2, 2], **kwargs)
    load_resnet_imagenet_pre_trained(model, MODEL_DIR + '/resnet18-5c106cde.pth')
    return model


def resnet34backbone(**kwargs):
    model = ResFeatExtractor(BasicBlock, [3, 4, 6, 3], **kwargs)
    load_resnet_imagenet_pre_trained(model, MODEL_DIR + '/resnet34-333f7ec4.pth')
    return model


def resnet50backbone(pre_trained=True, **kwargs):
    model = ResFeatExtractor(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pre_trained:
        pretrained = MODEL_DIR + '/resnet50-19c8e357.pth'
        model.load_state_dict(torch.load(pretrained), strict=False)
    return model


def resnet101backbone(pre_trained=True, **kwargs):
    model = ResFeatExtractor(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pre_trained:
        pretrained = MODEL_DIR + '/resnet101-5d3b4d8f.pth'
        model.load_state_dict(torch.load(pretrained), strict=False)
    return model


def resnet152backbone(pre_trained=True, **kwargs):
    model = ResFeatExtractor(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pre_trained:
        pretrained = MODEL_DIR + '/resnet152-b121ed2d.pth'
        model.load_state_dict(torch.load(pretrained), strict=False)
    return model


if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))