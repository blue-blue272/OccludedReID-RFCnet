from __future__ import print_function
from __future__ import division

import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

import data_manager
from dataset_loader import ImageDataset, ImageDataset_hardSplit_seg
import spatial_transforms as ST
import models
from losses import CrossEntropyLabelSmooth, TripletLoss, HardSplitLoss
from losses import MaskLoss, ClusterLoss
from utils.iotools import save_checkpoint, check_isfile
from utils.avgmeter import AverageMeter
from utils.logger import Logger
from utils.torchtools import count_num_param
from eval_metrics import evaluate
from samplers import RandomIdentitySampler
from optimizers import init_optim


parser = argparse.ArgumentParser(description='Train image model with cross entropy loss and hard triplet loss')
# Datasets
parser.add_argument('--root', type=str, default='', 
                    help="the path to dataset")
parser.add_argument('--fore_dir', type=str, default='', 
                    help="the path to extracted foremaps")
parser.add_argument('-d', '--dataset', type=str, default='market1501',
                    choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=256,
                    help="height of an image (default: 256)")
parser.add_argument('--width', type=int, default=128,
                    help="width of an image (default: 128)")
# Optimization options
parser.add_argument('--optim', type=str, default='adam',
                    help="optimization algorithm (see optimizers.py)")
parser.add_argument('--max-epoch', default=80, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--train-batch', default=64, type=int,
                    help="train batch size")
parser.add_argument('--test-batch', default=64, type=int,
                    help="test batch size")
parser.add_argument('--lr', '--learning-rate', default=0.00035, type=float,
                    help="initial learning rate")
parser.add_argument('--stepsize', default=[40, 60], nargs='+', type=int,
                    help="stepsize to decay learning rate")
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")
parser.add_argument('--weight-decay', default=5e-04, type=float,
                    help="weight decay (default: 5e-04)")
parser.add_argument('--num-instances', type=int, default=4,
                    help="number of instances per identity")
# Architecture
parser.add_argument('-a', '--arch', type=str, default='RFCnet', choices=models.get_names())
parser.add_argument('--save-dir', type=str, default='./result/market/RFCnet')
parser.add_argument('--resume', type=str, default='', metavar='PATH')

# foreground mask loss Lf
parser.add_argument('--alpha0', type=float, default=0.5,
                    help="the weight for the foreground mask loss")
# hard spatial attention loss LK
parser.add_argument('--alpha1', type=float, default=0.1,
                    help="the weight for the division split loss")
# region encoder appearance constrain loss LA
parser.add_argument('--alpha2', type=float, default=0.05,
                    help="the weight for the SFC encoder appearance constrain loss")
# region encoder position constrain loss LP
parser.add_argument('--alpha3', type=float, default=0.05,
                    help="the weight for the SFC encoder position constrain loss")
# Miscs
parser.add_argument('--distance', type=str, default='consine')
parser.add_argument('--seed', type=int, default=1,
                    help="manual seed")
parser.add_argument('--evaluate', action='store_true',
                    help="evaluation only")
parser.add_argument('--eval-step', type=int, default=10,
                    help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--start-eval', type=int, default=0,
                    help="start to evaluate after specific epoch")
parser.add_argument('--gpu-devices', default='0,1', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()


def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()

    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'), mode='a')
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'), mode='a')
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print("Initializing dataset {}".format(args.dataset))
    dataset = data_manager.init_imgreid_dataset(name=args.dataset, dataset_dir=args.root, fore_dir=args.fore_dir)

    transform_train = ST.Compose([
        ST.Scale((args.height, args.width), interpolation=3),
        ST.RandomHorizontalFlip(),
        ST.ToTensor(),
        ST.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ST.RandomErasing(0.5)
    ])

    transform_test = ST.Compose([
        ST.Scale((args.height, args.width), interpolation=3),
        ST.ToTensor(),
        ST.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pin_memory = True if use_gpu else False

    trainloader = DataLoader(
        ImageDataset_hardSplit_seg(dataset.train, transform=transform_train),
        sampler=RandomIdentitySampler(dataset.train, num_instances=args.num_instances),
        batch_size=args.train_batch, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True,
    )

    queryloader = DataLoader(
        ImageDataset(dataset.query, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    galleryloader = DataLoader(
        ImageDataset(dataset.gallery, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    print("Initializing model: {}".format(args.arch))
    model = models.init_model(name=args.arch, num_classes=dataset.num_train_pids) 
    print(model)

    criterion_xent = CrossEntropyLabelSmooth(use_gpu=use_gpu)
    criterion_htri = TripletLoss()
    criterion_mask = MaskLoss() 
    criterion_split = HardSplitLoss()
    criterion_cluster = ClusterLoss()
    
    optimizer = init_optim(args.optim, model.parameters(), args.lr, args.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.stepsize, gamma=args.gamma)

    if args.resume:
        if check_isfile(args.resume):
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            args.start_epoch = checkpoint['epoch']
            rank1 = checkpoint['rank1']
            print("Loaded checkpoint from '{}'".format(args.resume))
            print("- start_epoch: {}\n- rank1: {}".format(args.start_epoch, rank1))

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    if args.evaluate:
        print("Evaluate only")
        test(model, queryloader, galleryloader, use_gpu)
        return

    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0
    print("==> Start training")

    for epoch in range(args.start_epoch, args.max_epoch):

        start_train_time = time.time()
        train(epoch, model, criterion_xent, criterion_htri, criterion_mask, criterion_split, 
                criterion_cluster, optimizer, trainloader, use_gpu)
        train_time += round(time.time() - start_train_time)

        scheduler.step()

        
        if (epoch + 1) > args.start_eval and (epoch + 1) % args.eval_step == 0 or epoch == 0:
            print("==> Test")
            rank1 = test(model, queryloader, galleryloader, use_gpu)
            is_best = rank1 > best_rank1
            
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            
            save_checkpoint({
                'state_dict': state_dict,
                'rank1': rank1,
                'epoch': epoch,
            }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))

    print("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))
    print("==========\nArgs:{}\n==========".format(args))


def train(epoch, model, criterion_xent, criterion_htri, criterion_mask, criterion_split, criterion_cluster, optimizer, trainloader, use_gpu):
    xent_losses = AverageMeter()
    htri_losses = AverageMeter()
    mask_losses = AverageMeter()
    split_losses = AverageMeter()
    appearance_losses = AverageMeter()
    distance_losses = AverageMeter()
    accs = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()

    end = time.time()
    for batch_idx, (imgs, fore_masks, pids, _, _, split_param2, split_param3, _) in enumerate(trainloader):
        split_param2 = split_param2.long()
        split_param3 = split_param3.long()
        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()
            fore_masks = fore_masks.cuda()
            split_param2 = split_param2.cuda() #[bs, 4]
            split_param3 = split_param3.cuda() #[bs, 4]
            
        # measure data loading time
        data_time.update(time.time() - end)
            
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward
        outputs, features, a_fore, part_logit, distance_list = model(imgs)

        # combine hard triplet loss with cross entropy loss
        xent_loss = criterion_xent(outputs, pids)
        htri_loss = criterion_htri(features, pids)
        loss = xent_loss + htri_loss

        mask_loss1 = criterion_mask(a_fore[0], fore_masks)
        mask_loss2 = criterion_mask(a_fore[1], fore_masks)
        mask_loss = (mask_loss1 + mask_loss2) * 0.5

        split_loss1 = criterion_split(part_logit[0], split_param2)
        split_loss2 = criterion_split(part_logit[1], split_param3)
        split_loss = (split_loss1 + split_loss2) * 0.5

        appearance_loss1 = criterion_cluster(distance_list[0][0])
        appearance_loss2 = criterion_cluster(distance_list[1][0])
        appearance_loss = (appearance_loss1 + appearance_loss2) * 0.5

        distance_loss1 = criterion_cluster(distance_list[0][1])
        distance_loss2 = criterion_cluster(distance_list[1][1])
        distance_loss = (distance_loss1 + distance_loss2) * 0.5

        total_loss = loss + args.alpha0 * mask_loss +  args.alpha1 * split_loss + \
                args.alpha2 * appearance_loss + args.alpha3 * distance_loss

        # backward + optimize
        total_loss.backward()
        optimizer.step()

        # statistics
        _, preds = torch.max(outputs.data, 1)
        accs.update(torch.sum(preds == pids.data).float()/pids.size(0), pids.size(0))
        xent_losses.update(xent_loss.item(), pids.size(0))
        htri_losses.update(htri_loss.item(), pids.size(0))
        mask_losses.update(mask_loss.item(), pids.size(0))
        split_losses.update(split_loss.item(), pids.size(0))
        appearance_losses.update(appearance_loss.item(), pids.size(0))
        distance_losses.update(distance_loss.item(), pids.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch{0} '
          'Time:{batch_time.sum:.1f}s '
          'Data:{data_time.sum:.1f}s '
          'xentLoss:{xent_loss.avg:.4f} '
          'triLoss:{tri_loss.avg:.4f} '
          'MaskLoss:{mask_loss.avg:.4f} '
          'SplitLoss:{split_loss.avg:.4f} '
          'AppearanceLoss:{appearance_loss.avg:.4f} '
          'DistanceLoss:{distance_loss.avg:.4f} '
          'Acc:{acc.avg:.2%} '.format(
           epoch+1, batch_time=batch_time,
           data_time=data_time, xent_loss=xent_losses,
           tri_loss=htri_losses, mask_loss=mask_losses, 
           split_loss=split_losses, appearance_loss=appearance_losses,
           distance_loss=distance_losses, acc=accs))


def fliplr(img, use_gpu):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1, -1, -1).long()
    if use_gpu: inv_idx = inv_idx.cuda()
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def test(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20]):
    since = time.time()
    batch_time = AverageMeter()

    model.eval()
    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
            end = time.time()
            if use_gpu:
                imgs = imgs.cuda()

            n, c, h, w = imgs.size()
            features = torch.FloatTensor(n, model.module.feat_dim).zero_()
            for i in range(2):
                if (i==1):
                    imgs = fliplr(imgs, use_gpu)
                f = model(imgs)
                f = f.data.cpu()
                features = features + f

            batch_time.update(time.time() - end)

            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
            end = time.time()
            if use_gpu: 
                imgs = imgs.cuda()

            n, c, h, w = imgs.size()
            features = torch.FloatTensor(n, model.module.feat_dim).zero_()
            for i in range(2):
                if (i==1):
                    imgs = fliplr(imgs, use_gpu)
                f = model(imgs)
                f = f.data.cpu()
                features = features + f

            batch_time.update(time.time() - end)

            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
    
    print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, args.test_batch))

    m, n = qf.size(0), gf.size(0)
    distmat = torch.zeros((m, n))
    if args.distance == 'euclidean':
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
    else:
        q_norm = torch.norm(qf, p=2, dim=1, keepdim=True)
        g_norm = torch.norm(gf, p=2, dim=1, keepdim=True)
        qf = qf.div(q_norm.expand_as(qf))
        gf = gf.div(g_norm.expand_as(gf))
        distmat = - torch.mm(qf, gf.t())
    distmat = distmat.numpy()

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1]))
    print("------------------")

    return cmc[0]


if __name__ == '__main__':
    main()
