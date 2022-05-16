#!/usr/bin/env python3
import os
from argparse import ArgumentParser
import numpy as np
import torch
from data import get_dataset, DATASET_CONFIGS
from train import train
import utils
import config as cf
from networks import *

parser = ArgumentParser('EWC PyTorch Implementation')
parser.add_argument('--hidden-size', type=int, default=400)
parser.add_argument('--hidden-layer-num', type=int, default=6)
parser.add_argument('--hidden-dropout-prob', type=float, default=.5)
parser.add_argument('--input-dropout-prob', type=float, default=.2)

parser.add_argument('--task-number', type=int, default=8)
parser.add_argument('--epochs-per-task', type=int, default=10)
parser.add_argument('--lamda', type=float, default=40)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight-decay', type=float, default=0)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--test-size', type=int, default=128)
parser.add_argument('--adv_test_size', type=int, default=32)
parser.add_argument('--fisher-estimation-sample-size', type=int, default=1024)
parser.add_argument('--random-seed', type=int, default=0)
parser.add_argument('--no-gpus', action='store_false', dest='cuda')
parser.add_argument('--eval-log-interval', type=int, default=250)
parser.add_argument('--loss-log-interval', type=int, default=250)
parser.add_argument('--consolidate', action='store_true')
parser.add_argument('--with-scheduling',action='store_true')

#Resnet Parameters
parser.add_argument('--net_type', default='wide-resnet', type=str, help='model')
parser.add_argument('--depth', default=28, type=int, help='depth of model')
parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/cifar100]')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')

#PGD aparameters
parser.add_argument("--attack", type=str, default="pgd", choices=["fgsm", "pgd"])
parser.add_argument("--metric", type=str, default="Linf", choices=["Linf"])
parser.add_argument("--adv", type=str, default="none", choices=["none", "fgsm", "pgd", "ball"])
parser.add_argument("--eps", type=float, default=8.0)
parser.add_argument("--alpha", type=float, default=2.)
parser.add_argument("--iters", type=int, default=20)
parser.add_argument("--repeat", type=int, default=5)
parser.add_argument("--optim", type=str, default="sgd")


if __name__ == '__main__':
    args = parser.parse_args()

    # Use gpu is available
    cuda = torch.cuda.is_available() and args.cuda
    
    if(torch.cuda.is_available()):
        print("Cuda Available")

    np.random.seed(args.random_seed)

    # prepare datasets.
    if args.dataset == "cifar10":
        train_datasets = [
            get_dataset('cifar10_train')
        ]
        test_datasets = [
            get_dataset('cifar10_test')
        ]
    elif args.dataset == "cifar100":
        train_datasets = [
            get_dataset('cifar100_train')
        ]
        test_datasets = [
            get_dataset('cifar100_test')
        ]

    # prepare the model.
    if args.dataset == "cifar10":
        num_classes = 10
    elif args.dataset == "cifar100":
        num_classes = 100

    model = Wide_ResNet(args.depth, args.widen_factor, args.dropout, num_classes)
    file_name = 'wide-resnet-'+str(args.depth)+'x'+str(args.widen_factor)
    # initialize the parameters.
    checkpoint = torch.load('/w/247/diljot/pranav/checkpoint/'+args.dataset+os.sep+file_name+'.t7')
    model = checkpoint['net']


    # prepare the cuda if needed.
    if cuda:
        model.cuda()

    # run the experiment.
    train(
        model, train_datasets, test_datasets,
        epochs_per_task=args.epochs_per_task,
        batch_size=args.batch_size,
        test_size=args.test_size,
        consolidate=args.consolidate,
        fisher_estimation_sample_size=args.fisher_estimation_sample_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        eval_log_interval=args.eval_log_interval,
        loss_log_interval=args.loss_log_interval,
        cuda=cuda, adv_train = args.adv, args = args
    )
