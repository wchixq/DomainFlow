import os
import cv2
import numpy as np
import glob
import random

from config import ROOT_DIR


def generate_nonrepeative_random_ids(length, num):
    all_ids = np.arrange(stop=length, dtype=int)
    random.sample(population=all_ids, k=num)



# input shape: B*C, output shape: 1*C
def torch_calc_moment(input0, k=1, mean0=None):
    if mean0 is None:
        mean0 = input0.mean(dim=0)
    dev0 = input0 - mean0
    return (dev0**k).mean(dim=0)


def calculate_temporal_average(x, len0=5):
    len2plot = len(x)
    # number of left-alone points
    interval0 = len0 // 2

    smoothed = np.copy(x)
    for i in range(interval0, len2plot-interval0):
        smoothed[i] = np.mean(x[i-interval0:i+interval0+1])
    for i in range(1, interval0+1):
        smoothed[i] = np.mean(x[i - 1:i + 2])
    for i in range(len2plot-interval0, len2plot):
        smoothed[i] = np.mean(x[i - 1:i + 2])
    return smoothed


def set_random_seed(seed):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_image_label_list_file(list_file):
    img_list, labels = [], []
    f = open(list_file, 'r')
    lines = f.readlines()
    for line in lines:
        img_path, label = line.split()
        img_list.append(img_path)
        labels.append(label)
    f.close()
    return img_list, labels


def accuracy_np(probs, labels):
    labels = labels.data.cpu().numpy().reshape(-1).astype('i4')
    probs = probs.data.cpu().numpy()
    indices = np.argmax(probs, axis=1).reshape(-1).astype('i4')
    correct = np.sum(indices == labels)
    correct /= len(labels)
    return correct


def accuracy_torch(probs, labels):
    # multi-label accuracy
    labels = labels.data
    correct = labels.data - probs.data
    correct = correct.abs()
    correct = correct < 0.3
    correct = float(correct.sum())
    num_labels = float(labels.size(0) * labels.size(1))
    correct /= num_labels

    return correct


def acc_class_avg_numpy(preds, labels, num_class):
    acc = np.zeros(num_class, dtype=float)
    for n in range(num_class):
        ids = labels == n
        num_samples = len(ids)
        correct = np.count_nonzero(preds[ids] == labels[ids])
        acc[n] = float(correct) / float(num_samples)
    return acc


def acc_class_avg_torch(preds, labels, num_class):
    acc = np.zeros(num_class, dtype=float)
    for n in range(num_class):
        ids = labels == n
        num_samples = len(ids)
        correct = preds[ids].eq(labels[ids]).sum().cpu()
        acc[n] = float(correct) / float(num_samples)
    return acc


def log_no_print(writer, text):
    writer.write('%s\n' % text)
    writer.flush()


def log_with_print(writer, text):
    print(text)
    writer.write('%s\n' % text)



if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))
