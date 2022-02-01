## dependecies 
import argparse
import os
import shutil
import time
import torch
import torch.nn as nn

import models
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torchvision
from DataLoader_train import load_dataset
#from DataLoader_letters import load_torch_data_emnist
from tqdm import tqdm_notebook as tqdm
from loss import CrossEntropyLoss, mixup, TripletLoss, ContrastiveLoss1, SupConLoss
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from utils import *
from helper_functions import*
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import gzip, pickle
import matplotlib.cm as cm
import random
import math
from tsne import bh_sne
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser(description='PyTorch Omniglot')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('-c', '--checkpoint', default='pretrain_checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: pretrain_checkpoint)')
parser.add_argument('-d', '--save', default='savefiles_pretrain', type=str, metavar='PATH',
                    help='path to save files (default: save_files)')
parser.add_argument('-s', '--savefiles', default='savefiles_pretrain', type=str, metavar='PATH',
                    help='path to save files (default: save_files)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')


best_prec1 = 0

# Devices
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    global args, best_prec1
    args = parser.parse_args()
    history = TrainingHistory()

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.save+'/'+str(args.epochs)+'/'+args.checkpoint)

    if not os.path.isdir(args.savefiles):
        mkdir_p(args.save+'/'+str(args.epochs)+'/'+args.savefiles)

    

    model = models.Net(int(args.save)).to(device)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    extractor_params = list(map(id, model.extractor.parameters()))
    classifier_params = filter(lambda p: id(p) not in extractor_params, model.parameters())

    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    optimizer = torch.optim.Adamax([
                {'params': model.extractor.parameters()},
                {'params': classifier_params, 'lr': args.lr * 1}
            ], lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # optionally resume from a checkpoint
    title = 'Omniglot'
    if args.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.save+'/'+args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.save+'/'+args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.save+'/'+str(args.epochs)+'/'+args.resume, checkpoint['epoch']))
        logger = Logger(os.path.join(args.save+'/'+str(args.epochs)+'/'+args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.save+'/'+str(args.epochs)+'/'+args.resume, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # Data loading code
    batch_size = 64
    train_loader, val_loader = load_dataset(batch_size)


    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        lr = optimizer.param_groups[1]['lr']
        #lr = args.lr
        print('\nEpoch: [%d | %d]' % (epoch + 1, args.epochs))
        # train for one epoch
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        test_loss, test_acc, conf_matrix, feat_fc1 = validate(val_loader, model, criterion)

        scheduler.step()

        history.append(train_loss, test_loss, train_acc, test_acc)

        # append logger file
        logger.append([lr, train_loss, test_loss, train_acc, test_acc])

        # remember best prec@1 and save checkpoint
        is_best = test_acc > best_prec1
        best_prec1 = max(test_acc, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, checkpoint=args.save+'/'+str(args.epochs)+'/'+args.checkpoint)

    logger.close()
    #logger.plot()
    #savefig(os.path.join(args.save+'/'+args.checkpoint, 'log.eps'))

    """


    # Plot the representations

    # from fc1
    feature_fc1 = feat_fc1[0].astype(np.float64)
    target1 = feat_fc1[1]

    # feature
    feature_fc1 = bh_sne(feature_fc1[-3856:])
    target1 = target1[-3856:]

    plt.rcParams['figure.figsize'] = 20, 20
    plt.scatter(np.reshape(feature_fc1[:, 0], -1), np.reshape(feature_fc1[:, 1], -1), c=np.reshape(target1,-1), cmap=plt.cm.get_cmap("jet", 30))
    plt.colorbar(ticks=range(30))
    plt.title('Validation: Latent Projection-fc1')
    plt.savefig(str(args.save)+'/'+str(args.epochs)+'/'+args.savefiles+'/fc1.png', bbox_inches='tight')
    #plt.show()
    plt.clf()
    
    """

    # plot confusion matrix
    test_acc = str(round(test_acc, 2))
    #plot_confusion_matric(conf_matrix,args.save+'/'+str(args.epochs)+'/'+args.savefiles,'fc3_after_relu',test_acc)

    # plot train and accuracy loss
    history.plot(str(args.save)+'/'+str(args.epochs)+'/'+args.savefiles)

    print('Best acc:')
    print(best_prec1)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    bar = Bar('Training', max=len(train_loader))

    for batch_idx, data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        #data = mixup(data[0], data[1].long(), alpha, base_class)
        input = data[0].to(device)
        label = data[1].to(device)
        #label = torch.max(label, 1)[1]

        #output 
        logit,_ = model(input)
        
        # loss    
        loss = criterion(logit, label)

        # prediction vib
        pred = logit.argmax(1)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(logit, label, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        model.weight_norm()
        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(train_loader),
                    data=data_time.val,
                    bt=batch_time.val,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    out_target = []
    out_data = []
    out_output = []
    feat_ = []

    class_loss = 0
    info_loss = 0
    loss_vib = 0
    total_num = 0

    base_class = 964

    class_correct = [0]*base_class
    class_total = [0]*base_class
    conf_matrix_vib = np.zeros((base_class, base_class))

    latent_fc1 = Proj_Latent()

    # switch to evaluate mode
    model.eval()
    bar = Bar('Testing ', max=len(val_loader))

    with torch.no_grad():
        end = time.time()
        for batch_idx, data in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            input = data[0].to(device)
            label = data[1].long().to(device)

            # Compute output
            output,feature1 = model(input)
            
            # Loss
            loss = criterion(output, label)

            # Prediction
            pred_VIB = output.argmax(1)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, label, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))


            # compare predictions to true label
            correct_tensor_vib = pred_VIB.eq(label.data.view_as(pred_VIB))
            correct_vib = np.squeeze(correct_tensor_vib.numpy()) if device == "cpu" else np.squeeze(correct_tensor_vib.cpu().numpy())

            for label_a, pred_a_, correct_a in zip(label, pred_VIB, correct_vib):
                class_correct[label_a] += correct_a
                class_total[label_a] += 1

                # update confusion matrix
                conf_matrix_vib[label_a][pred_a_] += 1

            data_feature_extractor_fc1 = latent_fc1.append(feature1,label)
            

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

             # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(val_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
            bar.next()
        bar.finish()
    return (losses.avg, top1.avg, conf_matrix_vib, data_feature_extractor_fc1)


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

if __name__ == '__main__':
    main()