import argparse
import os
import shutil
import sys
import time
import csv
import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from torch.autograd import Variable
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings('ignore')

from data import CIFDataTest as CIFData
from data import collate_test_pool as collate_pool
from model.cgcnn import CrystalGraphConvNet
from model.pyramid import PyramidModel


parser = argparse.ArgumentParser(description='Crystal gated neural networks')
parser.add_argument('modelpath', help='path to the trained model.')
parser.add_argument('cifpath', help='path to the directory of CIF files.')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--model', choices=['cgcnn', 'pyramid', 'pyramid2'],
                    default='pyramid', help='training model')
parser.add_argument('--task-name', default='test', help='task name')
parser.add_argument('--task', choices=['regression', 'classification'],
                    default='regression', help='complete a regression or '
                                                   'classification task (default: regression)')
parser.add_argument('--max-num-nbr', default='7,9,13', type=str, metavar='N',
                    help='The maximum number of neighbors while constructing the crystal graph')
parser.add_argument('--radius', default='7,8,13', type=str, metavar='N',
                    help='The cutoff radius for searching neighbors')


args = parser.parse_args(sys.argv[1:])

output_dir = os.path.join('runs', args.model, args.task, args.task_name)
data_dir = os.path.join('runs', 'datasets', 'test')
args.modelpath = os.path.join(output_dir, args.modelpath)

if not isinstance(args.max_num_nbr, list):
    args.max_num_nbr = [int(item) for item in args.max_num_nbr.split(',')]
if not isinstance(args.radius, list):
    args.radius = [int(item) for item in args.radius.split(',')]

if os.path.isfile(args.modelpath):
    print("=> loading model params '{}'".format(args.modelpath))
    model_checkpoint = torch.load(args.modelpath,
                                  map_location=lambda storage, loc: storage)
    model_args = argparse.Namespace(**model_checkpoint['args'])
    print("=> loaded model params '{}'".format(args.modelpath))
else:
    print("=> no model params found at '{}'".format(args.modelpath))

args.cuda = not args.disable_cuda and torch.cuda.is_available()

if model_args.task == 'regression':
    best_mae_error = 1e10
else:
    best_mae_error = 0.


def main():
    global args, model_args, best_mae_error

    # load data
    dataset = CIFData(args.cifpath, max_num_nbr=args.max_num_nbr, radius=args.radius, output_dir=data_dir, task=args.task)
    collate_fn = collate_pool
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.workers, collate_fn=collate_fn,
                             pin_memory=args.cuda)

    # build model
    structures, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    # nbr_fea_len = structures[1].shape[-1]
    nbr_fea_len = [item.shape[-1] for item in structures[1]]
    if 'cgcnn' in args.modelpath:
        model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                    atom_fea_len=model_args.atom_fea_len,
                                    n_conv=model_args.n_conv,
                                    h_fea_len=model_args.h_fea_len,
                                    n_h=model_args.n_h,
                                    classification=True if model_args.task == 'classification' else False)
    else:
        model = PyramidModel(orig_atom_fea_len, nbr_fea_len,
                                atom_fea_len=model_args.atom_fea_len,
                                n_conv=model_args.n_conv,
                                h_fea_len=model_args.h_fea_len,
                                n_h=model_args.n_h,
                                classification=True if model_args.task == 'classification' else False)

    if args.cuda:
        model.cuda()

    # obtain target value normalizer
    normalizer = Normalizer(torch.zeros(3))

    # optionally resume from a checkpoint
    if os.path.isfile(args.modelpath):
        print("=> loading model '{}'".format(args.modelpath))
        checkpoint = torch.load(args.modelpath,
                                map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        normalizer.load_state_dict(checkpoint['normalizer'])
        print("=> loaded model '{}' (epoch {}, validation {})"
              .format(args.modelpath, checkpoint['epoch'],
                      checkpoint['best_mae_error']))
    else:
        print("=> no model found at '{}'".format(args.modelpath))

    validate(test_loader, model, normalizer, test=True)


def validate(val_loader, model, normalizer, test=False):
    batch_time = AverageMeter()
    if test:
        test_preds = []
        test_cif_ids = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    print('Starting Evaluation!!!')
    for i, (input, batch_cif_ids) in enumerate(val_loader):
        with torch.no_grad():
            if args.cuda:
                input_var = (Variable(input[0].cuda(non_blocking=True)),
                            [Variable(item.cuda(non_blocking=True)) for item in input[1]],
                            [item.cuda(non_blocking=True) for item in input[2]],
                            [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
            else:
                input_var = (Variable(input[0]),
                            [Variable(item) for item in input[1]],
                            input[2],
                            input[3])

        # compute output
        output = model(*input_var)

        if args.task == 'regression':
            test_pred = normalizer.denorm(output.data.cpu())
        else:
            test_pred = output.exp().argmax(dim=1).data.cpu()

        test_preds += test_pred.view(-1).tolist()
        test_cif_ids += batch_cif_ids

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}] | Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time))

    with open(os.path.join('test_results_{}.csv'.format(args.model)), 'w') as f:
        writer = csv.writer(f)
        for cif_id, pred in zip(test_cif_ids, test_preds):
            writer.writerow((cif_id, pred))

class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()
