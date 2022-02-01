import argparse
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import models
from tqdm import tqdm_notebook as tqdm
from DataLoader import load_torch_data
from helper_functions import*
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import gzip, pickle
import matplotlib.cm as cm
import random
import math
from tsne import bh_sne
from ipywidgets import IntProgress


parser = argparse.ArgumentParser(description='PyTorch Omniglot Training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--dimension', default=512, type=int,
                    metavar='N', help='latent dimension (default: 512)')
parser.add_argument('-c', '--checkpoint', default='imprint_checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: imprint_checkpoint)')
parser.add_argument('-d', '--save', default='savefiles_imprint', type=str, metavar='PATH',
                    help='path to save files (default: save_files)')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to model (default: none)')
parser.add_argument('--random', action='store_true', help='whether use random novel weights')
parser.add_argument('--numsample', default=1, type=int,
                    metavar='N', help='number of novel sample (default: 1)')
parser.add_argument('--test-novel-only', action='store_true', help='whether only test on novel classes')
parser.add_argument('--aug', action='store_true', help='whether use data augmentation during training')
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')

best_prec1 = 0


# Devices
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = 974
base_class = 964
novel_class = 10
alpha = 1

def main():
    # save accuracy for later
    accuracy_save = []

    global args, best_prec1
    args = parser.parse_args()

    if not os.path.isdir(args.checkpoint):
        mkdir_p(str(args.dimension)+'/'+str(args.epochs)+'/'+args.save)
        #mkdir_p(args.save+'/'+args.checkpoint)

    if not os.path.isdir(args.checkpoint):
        mkdir_p(str(args.dimension)+'/'+str(args.epochs)+'/'+args.checkpoint)

    model = models.Net(int(args.dimension)).to(device)

    # WI from fc1 before relu
    print(" ")
    print('==> Reading from model checkpoint... WI from fc1 before relu')
    print(args.model)
    print('************************************')
    assert os.path.isfile(args.model), 'Error: no model checkpoint directory found!'
    checkpoint = torch.load(args.model)
    args.start_epoch = checkpoint['epoch']
    best_prec1 = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded model checkpoint '{}' (epoch {})"
            .format(args.model, checkpoint['epoch']))
    cudnn.benchmark = True

    # Data loading code # fc1 before relu
    cm_avg = []
    test_avg = []
    cm_avg = np.zeros((10, 10))

    for i in tqdm(range(10)):
        test, novel,_ = load_torch_data(args.numsample, base_class)
        novel_weight = imprint(novel, model, 0) 
        test_acc,cm_test,fe_fc1 = validate(test, model, 0)

        test_avg.append(test_acc)
        cm_avg = cm_avg + cm_test
    
    print(" ")
    # Accuracy- Test data [0-9]
    test_avg_acc = np.array(test_avg).mean()
    test_avg_acc = str(round(test_avg_acc,2))
    #print(test_avg_acc) 
    c_matrix = np.rint((cm_avg / 10))


    # from fc1 before relu
    feature_fc1 = fe_fc1[0].astype(np.float64)
    target1 = fe_fc1[1]

    # feature
    feature_fc1 = bh_sne(feature_fc1[:5000])
    target1 = target1[:5000]


    plt.rcParams['figure.figsize'] = 20, 20
    plt.scatter(np.reshape(feature_fc1[:, 0], -1), np.reshape(feature_fc1[:, 1], -1), c=np.reshape(target1,-1), cmap=plt.cm.get_cmap("jet", 10))
    plt.colorbar(ticks=range(10))
    plt.title('Testing: Latent Projection-fc1')
    plt.savefig(str(args.dimension)+'/'+str(args.epochs)+'/'+args.save +'/fc1.png', bbox_inches='tight') #str(args.save)+'/'+str(args.epochs)+'/'+args.savefiles+'
    #plt.show()
    plt.clf()

    # plot confusion matrix
    plot_confusion_matric(c_matrix, str(args.dimension)+'/'+str(args.epochs)+'/'+args.save, 'fc1_before_relu',test_avg_acc)

    
    #####
    # WI from fc1 after relu
    model = models.Net(int(args.dimension)).to(device)

    print(" ")
    print('==> Reading from model checkpoint... WI from fc1 after relu')
    assert os.path.isfile(args.model), 'Error: no model checkpoint directory found!'
    checkpoint = torch.load(args.model)
    args.start_epoch = checkpoint['epoch']
    best_prec1 = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded model checkpoint '{}' (epoch {})"
            .format(args.model, checkpoint['epoch']))
    cudnn.benchmark = True

    # Data loading code # fc1 after relu
    cm_avg = []
    test_avg = []
    cm_avg = np.zeros((10, 10))
    for i in tqdm(range(10)):
        test, novel,_ = load_torch_data(args.numsample, base_class)
        novel_weight = imprint(novel, model, 1) 
        test_acc,cm_test,_= validate(test, model, 1)

        test_avg.append(test_acc)
        cm_avg = cm_avg + cm_test

    print(" ")
    # Accuracy- Test data [0-9]
    test_avg_acc1 = np.array(test_avg).mean()
    test_avg_acc1 = str(round(test_avg_acc1))
    #print(test_avg_acc1) 
    c_matrix = np.rint((cm_avg / 10))

    # plot confusion matrix
    plot_confusion_matric(c_matrix, str(args.dimension)+'/'+str(args.epochs)+'/'+args.save, 'fc1_after_relu',test_avg_acc1)

    accuracy_save.extend([args.dimension,test_avg_acc,test_avg_acc1])
    np.save(str(args.dimension)+'/'+str(args.epochs)+'/'+args.save+'_'+str(args.dimension)+'_'+'acc.npy',accuracy_save)

    save_checkpoint({
            'state_dict': model.state_dict(),
            'best_prec1': test_acc,
        }, checkpoint=str(args.dimension)+'/'+str(args.epochs)+'/'+args.checkpoint)


def imprint(novel_loader, model, select):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # switch to evaluate mode
    model.eval()
    end = time.time()
    bar = Bar('Imprinting', max=len(novel_loader))
    with torch.no_grad():
        for batch_idx, data in enumerate(novel_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            input = data[0].to(device)
            target = data[1].long().to(device)

            # compute output
            feature1,feature0 = model.helper_extract(input)

            if select==0:    feature = feature0 # before fc1 relu
            if select==1:    feature = feature1 # after fc1 relu
            

            output = F.normalize(feature,p=2,dim=feature.dim()-1,eps=1e-12)
            
            
            if batch_idx == 0:
                output_stack = output
                target_stack = target
            else:
                output_stack = torch.cat((output_stack, output), 0)
                target_stack = torch.cat((target_stack, target), 0)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:}'.format(
                        batch=batch_idx + 1,
                        size=len(novel_loader),
                        data=data_time.val,
                        bt=batch_time.val,
                        total=bar.elapsed_td,
                        eta=bar.eta_td
                        )
            bar.next()
        bar.finish()
      
    
    new_weight = torch.zeros(base_class, int(args.dimension)) # num_feature

    for i in range(novel_class):
        tmp = output_stack[target_stack == (i + base_class)].mean(0) if not args.random else torch.randn(10)
        new_weight[i] = tmp / tmp.norm(p=2)
        #new_weight[i] = F.normalize(tmp, p=2, dim=tmp.dim()-1, eps=1e-12)
 
    weight = torch.cat((model.classifier.fc.weight.data[:base_class], new_weight.to(device)))
    model.classifier.fc = nn.Linear(int(args.dimension), classes, bias=False)
    model.classifier.fc.weight.data = weight


   
def validate(val_loader,model,select):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    out_target = []
    out_data = []
    out_output = []
    feat_ = []

    class_correct = [0]*10
    class_total = [0]*10
    conf_matrix = np.zeros((10, 10))

    
    latent_fc1 = Proj_Latent()

    # switch to evaluate mode
    model.eval()
    bar = Bar('Testing   ', max=len(val_loader))
    with torch.no_grad():
        end = time.time()
        for batch_idx, data in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            input = data[0].to(device)
            label = data[1].long().to(device)

            # compute output
            if select==0:   output,feature1 = model.forward_wi_fc1(input)
            if select==1:   output,feature1 = model.forward_wi_fc1_(input)

            #predict --> convert output probabilities to predicted class
            pred = output.argmax(1)
           

            # measure accuracy
            prec1, prec5 = accuracy(output, label, topk=(1, 5))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))


            # compare predictions to true label
            correct_tensor = pred.eq(label.data.view_as(pred))
            correct = np.squeeze(correct_tensor.numpy()) if device == "cpu" else np.squeeze(correct_tensor.cpu().numpy())

            #conf_matrix = c_matrix(label, pred, correct)
            for label_, pred_, correct_ in zip(label, pred, correct):
                label_ -=964
                class_correct[label_] += correct_
                class_total[label_] += 1
                if pred_.data >=964:
                  # update confusion matrix
                  conf_matrix[label_][pred_-964] += 1
           
            
            # project latent representations
            data_feature_extractor_fc1= latent_fc1.append(feature1,label)


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

             # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(val_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
            bar.next()
        bar.finish()
    return top1.avg, conf_matrix, data_feature_extractor_fc1



def validate1(val_loader_test, model, select):
    top1 = AverageMeter()

    out_target = []
    out_data = []
    out_output = []
    feat_ = []

    class_correct = [0]*10
    class_total = [0]*10
    conf_matrix = np.zeros((10, 10))

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader_test):
            input = data[0].to(device)
            label = data[1].long().to(device)

            # compute output
            if select==1:   output,_, _ = model.forward_wi_fc1_(input)
            if select==2:   output,_, _ = model.forward_wi_fc2(input)
            if select==3:   output,_, _ = model.forward_wi_fc2_(input)

            #predict --> convert output probabilities to predicted class
            pred = output.argmax(1)
            
                   
            # measure accuracy
            prec1, _ = accuracy(output, label, topk=(1, 5))
            top1.update(prec1.item(), input.size(0))

            # compare predictions to true label
            correct_tensor = pred.eq(label.data.view_as(pred))
            correct = np.squeeze(correct_tensor.numpy()) if device == "cpu" else np.squeeze(correct_tensor.cpu().numpy())

            #conf_matrix = c_matrix(label, pred, correct)
            for label_, pred_, correct_ in zip(label, pred, correct):
                label_ -=964
                class_correct[label_] += correct_
                class_total[label_] += 1
                if pred_.data >=964:
                  # update confusion matrix
                  conf_matrix[label_][pred_-964] += 1
           
            
            # project latent representations
            #data_feature_extractor_test = latent_test.append(feature2,label)
            #data_feature_extractor_test_wi = latent_test_wi.append(feature1,label)

            
            # print progress
            if batch_idx == len(val_loader_test):
              print('({batch}/{size}) | top1: {top1: .4f} '.format(
                          batch=batch_idx + 1,
                          size=len(val_loader_test),
                          top1=top1.avg,
                          ))
              
    return top1.avg, conf_matrix


def save_checkpoint(state, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

if __name__ == '__main__':
    main()