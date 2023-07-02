import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from net.resnet_aap import resnet50backbone
from config import ROOT_DIR
MODEL_DIR = ROOT_DIR + '/models'


# Classifier adopted from MCD code ---------------------------------------------------------------
# https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
from torch.autograd import Function
class GradReverse(Function):
    # in Pytorch 1.3, customized function must be static, otherwise warnings
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output*-ctx.lambd), None


class ResClassifier(nn.Module):
    def __init__(self, channels=(2048, 256, 12), lambd=1.0, temperature=1.0):
        super().__init__()
        self.lambd = lambd
        layers = []
        for i in range(len(channels)-2):
            layers.append(nn.Dropout(p=0.5))
            layers.append(nn.Linear(channels[i], channels[i+1], bias=False))
            layers.append(nn.BatchNorm1d(channels[i+1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(channels[-2], channels[-1]))
        self.classifier = nn.Sequential(*layers)
        self.temperature = temperature

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward_reverse(self, x):
        x = GradReverse().apply(x, self.lambd)
        x = self.classifier(x)
        x = x / self.temperature
        return x

    def forward(self, x):
        x = self.classifier(x)
        x = x / self.temperature
        return x

    def set_train_mode(self):
        self.classifier.train()

    def set_eval_mode(self):
        self.classifier.eval()

    def move2cuda(self):
        self.classifier.cuda()

    def get_statedict_fromgpu(self):
        states = self.classifier.cpu().state_dict()
        self.classifier.cuda()
        return states

    def load_statedict(self, dict_name):
        self.classifier.load_state_dict(dict_name)

    def get_params(self):
        return self.classifier.parameters()


# MDD classifier ------------------------------------------------------------
# Adapted from thuml-MDD code
class GradientReverseLayer(Function):
    @staticmethod
    def forward(ctx, input, iter_count):
        ctx.iter_num = iter_count
        return input * 1.0

    @staticmethod
    def backward(ctx, grad_output):
        # coeff: [0.462*high_val+0.538*low_val, high_val]
        ctx.coeff = torch.tensor(2.0 * 0.1 / (1.0 + math.exp(-1.0 * ctx.iter_num / 1000.0)) - 0.1, device='cuda')
        return -ctx.coeff * grad_output, None


class Deconv2Net(nn.Module):
    def __init__(self, channels=(256, 128, 3)):
        super().__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.deconv1 = nn.ConvTranspose2d(channels[0], channels[1], kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(channels[1])
        self.deconv2 = nn.ConvTranspose2d(channels[1], channels[-1], kernel_size=2, stride=2)

    def forward(self, x):
        x = self.deconv1(x)
        x = self.bn2(x)
        x = self.deconv2(x)
        x = torch.sigmoid(x)  # sigmoid for normalized image generation
        return x


class FC2Classifier(nn.Module):
    def __init__(self, channels=(2048, 256, 12), temperature=1.0):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(channels[0], channels[1]),
            nn.BatchNorm1d(channels[1], affine=True),
            nn.ReLU(inplace=True))
        self.fc2 = nn.Linear(channels[1], channels[-1])
        self.temperature = temperature

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = x / self.temperature
        return x

    def forward_with_feat(self, x):
        x = self.fc1(x)
        x1 = self.fc2(x)
        x1 = x1 / self.temperature
        return x1, x

    def forward_fc2(self, x_fc1):
        x_fc1 = self.fc2(x_fc1)
        x_fc1 = x_fc1 / self.temperature
        return x_fc1


class MLPClassifier(nn.Module):
    def __init__(self, channels=(2048, 256, 12), temperature=1.0):
        super().__init__()
        layers = []
        for i in range(len(channels)-2):
            layers.append(nn.Linear(channels[i], channels[i+1], bias=False))
            layers.append(nn.BatchNorm1d(channels[i+1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(channels[-2], channels[-1]))
        self.classifier = nn.Sequential(*layers)
        self.temperature = temperature

    def forward(self, x):
        x = self.classifier(x)
        x = x / self.temperature
        return x


if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))