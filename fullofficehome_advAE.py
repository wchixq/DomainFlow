import os
from datetime import datetime
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.sampler import SequentialSampler, RandomSampler
import torch.nn.functional as F
import torch.optim as optim

from net.resnet_aap import resnet50backbone
from net.ellen_net import Deconv2Net, FC2Classifier, FC2RevClassifier
from utils.torch_net_utils import weights_init, throw_net_grad
from utils.torch_data_utils import log_with_print
from officehome import OfficeHomeDataset


def train(args):
    # set cuda device ----------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    init_rate = args.lr  # 0.001
    cls_lr = args.cls_lr
    num_epochs = args.num_epoch
    batch_size = args.batch_size
    fc_dim = args.fc_dim
    w_recon = args.weight_recon
    w_latent = args.weight_latent
    source = args.source
    target = args.target
    source_domains, target_domains = source.split(','), target.split(',')

    today = str(datetime.date(datetime.now()))
    year_month_date = today.split('-')
    date_to_save = year_month_date[0][2:] + year_month_date[1] + year_month_date[2]
    from config import ROOT_DIR, MACHINE
    save_dir = ROOT_DIR + '/results/%s_flowAE_fullofficehome' % date_to_save
    dataset_root = ROOT_DIR + '/datasets'
    exp_name = 'AdvAE%.2f_disc%.2f_seed%d_' % (w_recon,w_latent,seed) + source[0] + '2' + target[0]
    save_log_file = save_dir + '/train_%s_%s.txt' % (exp_name, MACHINE)
    os.makedirs(save_dir) if not os.path.exists(save_dir) else None
    img_resize, img_size = 256, 224
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(img_resize),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    test_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(img_resize),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    train_src_set = OfficeHomeDataset(dataset_root, source_domains, transforms=train_transforms)
    train_tgt_set = OfficeHomeDataset(dataset_root, target_domains, transforms=train_transforms)
    test_set = OfficeHomeDataset(dataset_root, target_domains, transforms=test_transforms)
    train_size, test_size = len(train_src_set), len(test_set)
    num_class = test_set.num_class
    num_batches = min(train_size // batch_size, test_size // batch_size)
    train_src_loader = DataLoader(train_src_set,
                               sampler=RandomSampler(train_src_set),
                               batch_size=batch_size,
                               drop_last=True,
                               num_workers=4,
                               pin_memory=True)
    train_tgt_loader = DataLoader(train_tgt_set,
                               sampler=RandomSampler(train_tgt_set),
                               batch_size=batch_size,
                               drop_last=True,
                               num_workers=4,
                               pin_memory=True)
    test_loader = DataLoader(test_set,
                               sampler=SequentialSampler(test_set),
                               batch_size=batch_size,
                               drop_last=False,
                               num_workers=4,
                               pin_memory=True)

    # net -----------------------------------------------------------------------------
    encoder = resnet50backbone().cuda().train()
    decoder = Deconv2Net(channels=(256, 128, 3)).cuda().train()
    classifier = FC2Classifier(channels=(2048, fc_dim, num_class)).cuda().train()
    discriminator = FC2RevClassifier(channels=(2048, 32, 2)).cuda().train()
    best_encoder_model = encoder.state_dict()
    best_decoder_model = decoder.state_dict()
    best_classifier_model = classifier.state_dict()
    best_discriminator_model = discriminator.state_dict()
    # use 10 times smaller lr for imagenet pre-trained parameters
    params_pretrained = list(encoder.parameters()) + list(discriminator.parameters())
    params_new = list(decoder.parameters()) + list(classifier.parameters())
    optimizer = optim.SGD([{'params': params_pretrained},
                           {'params': params_new, 'lr': init_rate*cls_lr}],
                          lr=init_rate, weight_decay=0.0005, momentum=0.9)
    criterion_cls = nn.CrossEntropyLoss().cuda()
    net_struct = resnet50backbone(pre_trained=True)  # automatically load ImageNet weights
    throw_net_grad(net_struct)
    net_struct.cuda().eval()
    crit_struct = nn.MSELoss().cuda()
    criterion_kl = nn.KLDivLoss().cuda()

    # --------------------------------------------------------------------
    l_ce_val, l_recon_val, l_latent_val, best_tgt_acc = 0.0, 0.0, 0.0, 0.0
    # source: 0, target: 1
    domain_labels = torch.zeros(batch_size*2, device='cuda').int().long()
    domain_labels[batch_size:].fill_(1)
    log_writer = open(save_log_file, 'w')
    time_now = str(datetime.time(datetime.now()))[:8]
    log_with_print(log_writer, 'Start time: %s (Date: %s)' % (time_now, date_to_save))
    log_with_print(log_writer, '***************************')
    log_with_print(log_writer, 'Exp: %s (on %s gpu%s)' % (exp_name, MACHINE, args.device))
    log_with_print(log_writer, '***************************')
    log_with_print(log_writer, 'Random seed: %d' % seed)
    log_with_print(log_writer, 'pre-trained: encoder')
    log_with_print(log_writer, 'learning rate pre-trained: %f' % init_rate)
    log_with_print(log_writer, 'params new: classifier, decoder')
    log_with_print(log_writer, 'learning rate new: %f' % (init_rate*cls_lr))
    log_with_print(log_writer, 'fc_dim: %d' % fc_dim)
    log_with_print(log_writer, 'Weight of reconstruction loss: %f' % w_recon)
    log_with_print(log_writer, 'Weight of latent regularization loss: %f' % w_latent)
    log_with_print(log_writer, 'Train size: %d Test size: %d Batch size: %d' % (train_size, test_size, batch_size))
    log_with_print(log_writer, 'Start training on %d batches...' % (num_batches))
    log_writer.flush()
    start = time.time()
    for epoch in range(1, num_epochs+1):
        it = 1
        encoder.train()
        classifier.train()
        for (img_src, label_s, _), (img_tgt, _, _) in zip(train_src_loader, train_tgt_loader):
            img_src, label_s = img_src.cuda(), label_s.cuda()
            img_tgt = img_tgt.cuda()
            optimizer.zero_grad()
            imgs = torch.cat((img_src, img_tgt), dim=0)
            out_both, l1feat_both = encoder.forward_withl1feat(imgs)
            recon_both = decoder(l1feat_both)
            both_logit = classifier(out_both)

            domains = discriminator.forward_train(out_both)
            l_latent = criterion_cls(domains, domain_labels)

            l_ce = criterion_cls(both_logit[:batch_size], label_s)
            # reconstruction loss at high-level feature
            l4_out_recon = net_struct.forward_to_layer4(recon_both)
            l4_img_origin = net_struct.forward_to_layer4(imgs)
            l_recon = crit_struct(l4_out_recon, l4_img_origin.detach())

            l_total = l_ce + w_recon*l_recon + w_latent*l_latent
            l_total.backward()
            optimizer.step()
            if it == num_batches:
                l_ce_val = float(l_ce.item())
                l_recon_val = float(l_recon.item())
                l_latent_val = float(l_latent.item())
                break
            it += 1
        encoder.eval()
        classifier.eval()
        correct = 0.0
        with torch.no_grad():
            for image, label, _ in test_loader:
                image, label = image.cuda(), label.cuda()
                output = classifier(encoder(image))
                pred = output.data.max(1)[1]
                correct += float(pred.eq(label.data).cpu().sum())
        acc_ep = float(correct) / float(test_size) * 100.0
        if acc_ep > best_tgt_acc:
            best_tgt_acc = acc_ep
            best_encoder_model = encoder.state_dict()
            best_decoder_model = decoder.state_dict()
            best_classifier_model = classifier.state_dict()
            best_discriminator_model = discriminator.state_dict()
        time_taken = (time.time() - start) / 60.0
        log_with_print(log_writer, 'epoch%03d: l_ce:%f l_recon:%f l_lat:%f test_acc:%.2f/%.2f in%.1fmin' % (
            epoch, l_ce_val, l_recon_val, l_latent_val, acc_ep, best_tgt_acc, time_taken))
        log_writer.flush()
    time_taken = (time.time() - start) / 3600.0
    time_now = str(datetime.time(datetime.now()))[:8]
    today = str(datetime.date(datetime.now()))
    year_month_date = today.split('-')
    date_now = year_month_date[0][2:] + year_month_date[1] + year_month_date[2]
    log_with_print(log_writer, 'Finish time: %s (Date: %s)' % (time_now, date_now))
    log_with_print(log_writer, '\nBest target accuracy: %.2f' % best_tgt_acc)
    log_with_print(log_writer, 'Total time taken: %.1f hrs' % time_taken)
    log_writer.close()
    torch.save(best_encoder_model, save_dir + '/%s_%.2f_final_encoder.pth' % (exp_name, best_tgt_acc))
    torch.save(best_decoder_model, save_dir + '/%s_%.2f_final_decoder.pth' % (exp_name, best_tgt_acc))
    torch.save(best_classifier_model, save_dir + '/%s_%.2f_final_classifier.pth' % (exp_name, best_tgt_acc))
    torch.save(best_discriminator_model, save_dir + '/%s_%.2f_final_discriminator.pth' % (exp_name, best_tgt_acc))
    print('Finish training')


if __name__ == '__main__':
    RUN_FILE = os.path.basename(__file__)
    print('running %s...' % RUN_FILE)
    # check_dataset()
    import argparse
    parser = argparse.ArgumentParser(description='OfficeHome DA')
    parser.add_argument('--device', type=str, default='0', metavar='TS', help='Domain ID')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='Domain ID')
    parser.add_argument('--num_epoch', type=int, default=60, metavar='NE', help='Domain ID')
    parser.add_argument('--batch_size', type=int, default=32, metavar='BS', help='Domain ID')
    parser.add_argument('--source', type=str, default='Product', metavar='SR', help='Domain ID')
    parser.add_argument('--target', type=str, default='RealWorld', metavar='TS', help='Domain ID')
    parser.add_argument('--weight_recon', type=float, default=1.0, metavar='TS', help='Domain ID')
    parser.add_argument('--cls_lr', type=float, default=1.0)
    parser.add_argument('--fc_dim', type=int, default=512)
    parser.add_argument('--weight_latent', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=202)
    args = parser.parse_args()
    train(args)