import os
import torch
# import torch_xla
# import torch_xla.debug.metrics as met
# import torch_xla.distributed.data_parallel as dp
# import torch_xla.distributed.parallel_loader as pl
# import torch_xla.utils.utils as xu
# import torch_xla.core.xla_model 
# import torch_xla.utils.serialization as xser
# import torch_xla.distributed.xla_multiprocessing as xmp
# import torch_xla.test.test_utils as test_utils
# from warmup_scheduler import GradualWarmupScheduler
import sys; 
package_paths = [
    '../input/pytorch-image-models/pytorch-image-models-master',
    '../input/image-fmix/FMix-master'
]


for pth in package_paths:
    sys.path.append(pth)
    
import warnings
import pandas as pd
import numpy as np
import torch.nn as nn
# from sklearn.model_selection import train_test_split
from sklearn import metrics
# from transformers import get_linear_schedule_with_warmup
import time
import torchvision
import torch.nn as nn
from tqdm import tqdm_notebook as tqdm
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import torch.optim as optim
from torchvision import transforms
from torch.optim import lr_scheduler
import sys
import gc
import os
import random
import skimage.io
from PIL import Image
import scipy as sp
import sklearn.metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from functools import partial
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from albumentations import Compose, Normalize, HorizontalFlip, VerticalFlip
from albumentations.pytorch import ToTensorV2
from contextlib import contextmanager
from pathlib import Path
from collections import defaultdict, Counter
from fmix import sample_mask, make_low_freq_image, binarise_mask
from glob import glob
from sklearn.model_selection import GroupKFold, StratifiedKFold
import cv2
from skimage import io
import torch
from torch import nn
import os
from datetime import datetime
import time
import random

import torchvision
from torchvision import transforms
import pandas as pd
import numpy as np
from tqdm import tqdm


import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.cuda.amp import autocast, GradScaler
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F

import timm

import sklearn

from sklearn.metrics import roc_auc_score, log_loss
import cv2
from scipy.ndimage.interpolation import zoom
from dataloader import prepare_dataloader


warnings.filterwarnings("ignore")

print(torch.__version__)

CFG = {
    'fold_num': 5,
    'seed': 719,
    'model_arch': 'vit_base_patch16_384',
    'img_size': 384,
    'epochs': 10,
    'train_bs': 10,
    'valid_bs': 10,
    'T_0': 10,
    'lr': 1e-4,
    'min_lr': 1e-4, 
    'smoothing' : 0.06,
    't1' : 0.8,
    't2' : 1.4,
    'warmup_factor' : 7,
    'warmup_epo' : 1,
    'num_workers': 4,#8
    'accum_iter': 2, # batch accumulation for backprop with effectively larger batch size
    'verbose_step': 1
}


# Code taken from https://github.com/fhopfmueller/bi-tempered-loss-pytorch/blob/master/bi_tempered_loss_pytorch.py

def log_t(u, t):
    """Compute log_t for `u'."""
    if t==1.0:
        return u.log()
    else:
        return (u.pow(1.0 - t) - 1.0) / (1.0 - t)

def exp_t(u, t):
    """Compute exp_t for `u'."""
    if t==1:
        return u.exp()
    else:
        return (1.0 + (1.0-t)*u).relu().pow(1.0 / (1.0 - t))

def compute_normalization_fixed_point(activations, t, num_iters):

    """Returns the normalization value for each example (t > 1.0).
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature 2 (> 1.0 for tail heaviness).
      num_iters: Number of iterations to run the method.
    Return: A tensor of same shape as activation with the last dimension being 1.
    """
    mu, _ = torch.max(activations, -1, keepdim=True)
    normalized_activations_step_0 = activations - mu

    normalized_activations = normalized_activations_step_0

    for _ in range(num_iters):
        logt_partition = torch.sum(
                exp_t(normalized_activations, t), -1, keepdim=True)
        normalized_activations = normalized_activations_step_0 * \
                logt_partition.pow(1.0-t)

    logt_partition = torch.sum(
            exp_t(normalized_activations, t), -1, keepdim=True)
    normalization_constants = - log_t(1.0 / logt_partition, t) + mu

    return normalization_constants

def compute_normalization_binary_search(activations, t, num_iters):

    """Returns the normalization value for each example (t < 1.0).
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature 2 (< 1.0 for finite support).
      num_iters: Number of iterations to run the method.
    Return: A tensor of same rank as activation with the last dimension being 1.
    """

    mu, _ = torch.max(activations, -1, keepdim=True)
    normalized_activations = activations - mu

    effective_dim = \
        torch.sum(
                (normalized_activations > -1.0 / (1.0-t)).to(torch.int32),
            dim=-1, keepdim=True).to(activations.dtype)

    shape_partition = activations.shape[:-1] + (1,)
    lower = torch.zeros(shape_partition, dtype=activations.dtype, device=activations.device)
    upper = -log_t(1.0/effective_dim, t) * torch.ones_like(lower)

    for _ in range(num_iters):
        logt_partition = (upper + lower)/2.0
        sum_probs = torch.sum(
                exp_t(normalized_activations - logt_partition, t),
                dim=-1, keepdim=True)
        update = (sum_probs < 1.0).to(activations.dtype)
        lower = torch.reshape(
                lower * update + (1.0-update) * logt_partition,
                shape_partition)
        upper = torch.reshape(
                upper * (1.0 - update) + update * logt_partition,
                shape_partition)

    logt_partition = (upper + lower)/2.0
    return logt_partition + mu

class ComputeNormalization(torch.autograd.Function):
    """
    Class implementing custom backward pass for compute_normalization. See compute_normalization.
    """
    @staticmethod
    def forward(ctx, activations, t, num_iters):
        if t < 1.0:
            normalization_constants = compute_normalization_binary_search(activations, t, num_iters)
        else:
            normalization_constants = compute_normalization_fixed_point(activations, t, num_iters)

        ctx.save_for_backward(activations, normalization_constants)
        ctx.t=t
        return normalization_constants

    @staticmethod
    def backward(ctx, grad_output):
        activations, normalization_constants = ctx.saved_tensors
        t = ctx.t
        normalized_activations = activations - normalization_constants 
        probabilities = exp_t(normalized_activations, t)
        escorts = probabilities.pow(t)
        escorts = escorts / escorts.sum(dim=-1, keepdim=True)
        grad_input = escorts * grad_output
        
        return grad_input, None, None

def compute_normalization(activations, t, num_iters=5):
    """Returns the normalization value for each example. 
    Backward pass is implemented.
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
      num_iters: Number of iterations to run the method.
    Return: A tensor of same rank as activation with the last dimension being 1.
    """
    return ComputeNormalization.apply(activations, t, num_iters)

def tempered_sigmoid(activations, t, num_iters = 5):
    """Tempered sigmoid function.
    Args:
      activations: Activations for the positive class for binary classification.
      t: Temperature tensor > 0.0.
      num_iters: Number of iterations to run the method.
    Returns:
      A probabilities tensor.
    """
    internal_activations = torch.stack([activations,
        torch.zeros_like(activations)],
        dim=-1)
    internal_probabilities = tempered_softmax(internal_activations, t, num_iters)
    return internal_probabilities[..., 0]


def tempered_softmax(activations, t, num_iters=5):
    """Tempered softmax function.
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature > 1.0.
      num_iters: Number of iterations to run the method.
    Returns:
      A probabilities tensor.
    """
    if t == 1.0:
        return activations.softmax(dim=-1)

    normalization_constants = compute_normalization(activations, t, num_iters)
    return exp_t(activations - normalization_constants, t)

def bi_tempered_binary_logistic_loss(activations,
        labels,
        t1,
        t2,
        label_smoothing = 0.0,
        num_iters=5,
        reduction='mean'):

    """Bi-Tempered binary logistic loss.
    Args:
      activations: A tensor containing activations for class 1.
      labels: A tensor with shape as activations, containing probabilities for class 1
      t1: Temperature 1 (< 1.0 for boundedness).
      t2: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
      label_smoothing: Label smoothing
      num_iters: Number of iterations to run the method.
    Returns:
      A loss tensor.
    """
    internal_activations = torch.stack([activations,
        torch.zeros_like(activations)],
        dim=-1)
    internal_labels = torch.stack([labels.to(activations.dtype),
        1.0 - labels.to(activations.dtype)],
        dim=-1)
    return bi_tempered_logistic_loss(internal_activations, 
            internal_labels,
            t1,
            t2,
            label_smoothing = label_smoothing,
            num_iters = num_iters,
            reduction = reduction)

def bi_tempered_logistic_loss(activations,
        labels,
        t1,
        t2,
        label_smoothing=0.0,
        num_iters=5,
        reduction = 'mean'):

    """Bi-Tempered Logistic Loss.
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      labels: A tensor with shape and dtype as activations (onehot), 
        or a long tensor of one dimension less than activations (pytorch standard)
      t1: Temperature 1 (< 1.0 for boundedness).
      t2: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
      label_smoothing: Label smoothing parameter between [0, 1). Default 0.0.
      num_iters: Number of iterations to run the method. Default 5.
      reduction: ``'none'`` | ``'mean'`` | ``'sum'``. Default ``'mean'``.
        ``'none'``: No reduction is applied, return shape is shape of
        activations without the last dimension.
        ``'mean'``: Loss is averaged over minibatch. Return shape (1,)
        ``'sum'``: Loss is summed over minibatch. Return shape (1,)
    Returns:
      A loss tensor.
    """

    if len(labels.shape)<len(activations.shape): #not one-hot
        labels_onehot = torch.zeros_like(activations)
        labels_onehot.scatter_(1, labels[..., None], 1)
    else:
        labels_onehot = labels

    if label_smoothing > 0:
        num_classes = labels_onehot.shape[-1]
        labels_onehot = ( 1 - label_smoothing * num_classes / (num_classes - 1) ) \
                * labels_onehot + \
                label_smoothing / (num_classes - 1)

    probabilities = tempered_softmax(activations, t2, num_iters)

    loss_values = labels_onehot * log_t(labels_onehot + 1e-10, t1) \
            - labels_onehot * log_t(probabilities, t1) \
            - labels_onehot.pow(2.0 - t1) / (2.0 - t1) \
            + probabilities.pow(2.0 - t1) / (2.0 - t1)
    loss_values = loss_values.sum(dim = -1) #sum over classes

    if reduction == 'none':
        return loss_values
    if reduction == 'sum':
        return loss_values.sum()
    if reduction == 'mean':
        return loss_values.mean()
    
    

def get_probs(activations,
        labels,
        t1,
        t2,
        label_smoothing=0.0,
        num_iters=5,
        reduction = 'mean'):

    """Bi-Tempered Logistic Loss.
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      labels: A tensor with shape and dtype as activations (onehot), 
        or a long tensor of one dimension less than activations (pytorch standard)
      t1: Temperature 1 (< 1.0 for boundedness).
      t2: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
      label_smoothing: Label smoothing parameter between [0, 1). Default 0.0.
      num_iters: Number of iterations to run the method. Default 5.
      reduction: ``'none'`` | ``'mean'`` | ``'sum'``. Default ``'mean'``.
        ``'none'``: No reduction is applied, return shape is shape of
        activations without the last dimension.
        ``'mean'``: Loss is averaged over minibatch. Return shape (1,)
        ``'sum'``: Loss is summed over minibatch. Return shape (1,)
    Returns:
      A loss tensor.
    """

    if len(labels.shape)<len(activations.shape): #not one-hot
        labels_onehot = torch.zeros_like(activations)
        labels_onehot.scatter_(1, labels[..., None], 1)
    else:
        labels_onehot = labels

    if label_smoothing > 0:
        num_classes = labels_onehot.shape[-1]
        labels_onehot = ( 1 - label_smoothing * num_classes / (num_classes - 1) ) \
                * labels_onehot + \
                label_smoothing / (num_classes - 1)

    probabilities = tempered_softmax(activations, t2, num_iters)
    return probabilities


def train_one_epoch(epoch, model, optimizer, train_loader, device, scheduler=None, schd_batch_update=False):
    model.train()

    t = time.time()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    z = 0
    for step, (imgs, image_labels) in pbar:
        z = z + 1
        if z % 20 == 0:
            gc.collect()
        
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()

        
        image_preds = model(imgs)   #output = model(input)

        #loss = loss_fn(image_preds, image_labels)
        loss = bi_tempered_logistic_loss(image_preds, image_labels, t1=CFG['t1'], t2=CFG['t2'], label_smoothing=CFG['smoothing'])
        
        #scaler.scale(loss).backward()
        
        loss.backward()
        #scaler.step(optimizer)
        optimizer.step()
        #scaler.update()
        optimizer.zero_grad() 
        
        if scheduler is not None and schd_batch_update:
            scheduler.step()

        # if ((step + 1) %  CFG['accum_iter'] == 0) or ((step + 1) == len(train_loader)):
        #     # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)

        #     #scaler.step(optimizer)
        #     optimizer.step()
        #     #scaler.update()
        #     optimizer.zero_grad() 
            
        #     if scheduler is not None and schd_batch_update:
        #         scheduler.step()
    pbar.close()
    if scheduler is not None and not schd_batch_update:
        scheduler.step()
        
def valid_one_epoch(epoch, model, val_loader, device):
    model.eval()

    t = time.time()
    loss_sum = 0
    sample_num = 0
    image_preds_all = []
    image_targets_all = []

    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()
        
        image_preds = model(imgs)   #output = model(input)
        image_preds_all += [torch.argmax(image_preds, 1).detach().cpu().numpy()]
        image_targets_all += [image_labels.detach().cpu().numpy()]
        
        #loss = loss_fn(image_preds, image_labels)
        loss = bi_tempered_logistic_loss(image_preds, image_labels, t1=CFG['t1'], t2=CFG['t2'], label_smoothing=CFG['smoothing'])
        
        #loss_sum += loss*image_labels.shape[0]
        #sample_num += image_labels.shape[0]  

    pbar.close()
    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    acc = (image_preds_all==image_targets_all).mean()
    #LOGGER.debug('validation multi-class accuracy = {:.4f}'.format(acc))
    print(f"Validation Accuracy = {acc:.4f}")
   
    return loss, acc

class CassvaImgClassifier(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        #print(self.model)
        n_features = self.model.head.in_features
        self.model.head = nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x
    
model = CassvaImgClassifier(CFG['model_arch'], 5, pretrained=True)


def train_model(folds = range(0, 5)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    data = pd.read_csv('./data.csv')
    data = data.rename(columns={'target': 'label', 'img_name' : "image_id"})
    
    for fold in folds:
        trn_idx = data[data['fold'] != fold]
        val_idx = data[data['fold'] == fold]

        lr =  CFG['lr']
        optimizer = torch.optim.Adam(model.parameters(), lr=lr/CFG['warmup_factor'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG['T_0'], T_mult=1, eta_min=CFG['min_lr'], last_epoch=-1)

        best_accuracy = 0.0
        
        for epoch in range(CFG['epochs']):
            gc.collect()
            train_loader, val_loader = prepare_dataloader(trn_idx, val_idx, data_root='../input/cassava-leaf-disease-classification/train_images/')
            # para_loader = pl.ParallelLoader(train_loader, [device])
            train_one_epoch(epoch, model, optimizer, train_loader, device, scheduler=scheduler, schd_batch_update=False)
    
            
            del(train_loader)
            gc.collect()
            
            val_loss,cur_accuracy = valid_one_epoch(epoch, model, val_loader, device)
            
            del(val_loader)
            gc.collect()

            content = time.ctime() + ' ' + f'FOLD -> {fold} --> Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val_loss : {(val_loss):.5f},Validation_Accuracy: {(cur_accuracy):.5f}'

            with open(f'log.txt', 'a') as appender:
                appender.write(content + '\n')
            
            torch.save(model.state_dict(),'{}_fold_{}_best_epoch_{}_Validation_Accuracy.h5'.format(CFG['model_arch'], epoch,cur_accuracy))
            
             
            if cur_accuracy >= best_accuracy:
                torch.save(model.state_dict(),'{}_fold_{}_best_epoch'.format(CFG['model_arch'], fold))
                best_accuracy = cur_accuracy
                    
        
#important, we specify here the fold to train (one out of [0,4])
# folds_to_train  = [1]

# def _mp_fn(rank, flags):
#     global acc_list
#     torch.set_default_tensor_type('torch.FloatTensor')
#     res = train_model()

# FLAGS={}
# xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')

# gc.collect()
# torch.cuda.empty_cache()
# try:
#     del(model)
    
# except:
#     pass
# gc.collect()

# f = open(f'./log.txt', "r")

# print(f.read())

if __name__ == "__main__":
    res = train_model()