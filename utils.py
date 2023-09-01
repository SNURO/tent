import argparse
import json
import math
import os
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Union
import numpy as np
import time

import ipdb

import requests
import torch
from torch import nn

def oracle_accuracy(model: nn.Module,
                   x: torch.Tensor,
                   y: torch.Tensor,
                   batch_size: int = 100,
                   device: torch.device = None,
                   iteration = 1,
                   linear_retrain = False):
    if device is None:
        device = x.device
    criteria = nn.CrossEntropyLoss()
    optimizer = model.optimizer
    acc = 0.
    n_batches = math.ceil(x.shape[0] / batch_size)
    if linear_retrain:
        model = model.model
        optimizer.param_groups[0]['lr'] = 0.0002

    for counter in range(n_batches):
        x_curr = x[counter * batch_size:(counter + 1) *
                    batch_size].to(device)
        y_curr = y[counter * batch_size:(counter + 1) *
                    batch_size].to(device).long()
        
        for iter in range(iteration):
            output = model(x_curr)
            loss = criteria(output, y_curr)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            #if linear_retrain:
                #ipdb.set_trace()
            #ipdb.set_trace()
            if iter==iteration-1:
                acc += (output.max(1)[1] == y_curr).float().sum()

    return acc.item() / x.shape[0]

def oracle_accuracy_multi(model: nn.Module,
                   x: torch.Tensor,
                   y: torch.Tensor,
                   batch_size: int = 100,
                   device: torch.device = None,
                   linear_retrain = False,
                   epochs=1):
    if device is None:
        device = x.device
    criteria = nn.CrossEntropyLoss()
    optimizer = model.optimizer
    acc = 0.
    n_batches = math.ceil(x.shape[0] / batch_size)
    if linear_retrain:
        model = model.model
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.1

    for epoch in range(epochs):
        start=time.time()
        acc=0.
        last = False
        if epoch==epochs-1: last=True

        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) *
                        batch_size].to(device)
            y_curr = y[counter * batch_size:(counter + 1) *
                        batch_size].to(device).long()

            output = model(x_curr)
            loss = criteria(output, y_curr)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            #if linear_retrain:
                #ipdb.set_trace()
            #ipdb.set_trace()
            #ipdb.set_trace()

            acc += (output.max(1)[1] == y_curr).float().sum()
        end=time.time()
        print(f"epoch[{epoch}/{epochs}]      {end - start:.0f}s     {(1-acc.item()/x.shape[0])*100}")

    return acc.item() / x.shape[0]

def oracle_accuracy_split(model: nn.Module,
                   x_train: torch.Tensor,
                   y_train: torch.Tensor,
                   x_test: torch.Tensor,
                   y_test: torch.Tensor,
                   batch_size: int = 100,
                   device: torch.device = None,
                   linear_retrain = False,
                   epochs=1):
    if device is None:
        device = x_train.device
    criteria = nn.CrossEntropyLoss()
    optimizer = model.optimizer
    acc = 0.
    n_batches = math.ceil(x_train.shape[0] / batch_size)
    if linear_retrain:
        model = model.model
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001

    for epoch in range(epochs):
        start=time.time()
        acc=0.
        last = False
        if epoch==epochs-1: last=True

        for counter in range(n_batches):
            x_curr = x_train[counter * batch_size:(counter + 1) *
                        batch_size].to(device)
            y_curr = y_train[counter * batch_size:(counter + 1) *
                        batch_size].to(device).long()

            output = model(x_curr)
            loss = criteria(output, y_curr)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            #if linear_retrain:
                #ipdb.set_trace()
            #ipdb.set_trace()
            #ipdb.set_trace()

            acc += (output.max(1)[1] == y_curr).float().sum()
        end=time.time()
        print(f"epoch[{epoch}/{epochs}]      {end - start:.0f}s     {(1-acc.item()/x_train.shape[0])*100}")

        if epoch%10==0:
            val_acc = clean_accuracy(model, x_test, y_test, batch_size=batch_size, device=device)
            print(f"epoch[{epoch}/{epochs}]     val_acc       {(1-val_acc)*100}")

    return val_acc

def clean_accuracy(model: nn.Module,
                   x: torch.Tensor,
                   y: torch.Tensor,
                   batch_size: int = 100,
                   device: torch.device = None,
                   iteration = 1):
    if device is None:
        device = x.device
    acc = 0.
    n_batches = math.ceil(x.shape[0] / batch_size)
    with torch.no_grad():
        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) *
                       batch_size].to(device)
            y_curr = y[counter * batch_size:(counter + 1) *
                       batch_size].to(device)

            for iter in range(iteration):
                output = model(x_curr)
            acc += (output.max(1)[1] == y_curr).float().sum()

    return acc.item() / x.shape[0]

# for obtaining activation histogram (for masking protocol) : 보류
def get_histogram(model: nn.Module,
                   x: torch.Tensor,
                   y: torch.Tensor,
                   batch_size: int = 100,
                   device: torch.device = None,
                   iteration = 1):
    if device is None:
        device = x.device
    acc = 0.
    n_batches = math.ceil(x.shape[0] / batch_size)
    with torch.no_grad():
        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) *
                       batch_size].to(device)
            y_curr = y[counter * batch_size:(counter + 1) *
                       batch_size].to(device)

            for iter in range(iteration):
                output = model(x_curr)
            
            # activation map에서 histogram 정보 뽑아내서 누적시키기 

    return acc.item() / x.shape[0]

#get data_num X class_num tensor output from model (TENT의 그냥 forward call 하면 자동으로 수행되는 backward도 무시: .forward_only)
def get_stats(model: nn.Module,
                   x: torch.Tensor,
                   y: torch.Tensor,
                   batch_size: int = 100,
                   device: torch.device = None,
                   iteration = 1,
                   corruption_type = 'None',
                   return_raw = False, 
                   no_grad = True,
                   gt_label = False):
    if device is None:
        device = x.device
    all_outputs=torch.zeros((x.shape[0],10))
    acc = 0.
    n_batches = math.ceil(x.shape[0] / batch_size)

    if no_grad:
        with torch.no_grad():
            for counter in range(n_batches):
                x_curr = x[counter * batch_size:(counter + 1) *
                        batch_size].to(device)
                y_curr = y[counter * batch_size:(counter + 1) *
                        batch_size].to(device)
                try:
                    output = model.forward_only(x_curr)
                except:
                    output = model(x_curr)
                    warnings.warn("model has no forward_only method, using forward instead")
                acc += (output.max(1)[1] == y_curr).float().sum()

                all_outputs[counter * batch_size:(counter + 1) *
                        batch_size] = output.cpu()
    else:
        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) *
                    batch_size].to(device)
            y_curr = y[counter * batch_size:(counter + 1) *
                    batch_size].to(device)

            for iter in range(iteration):
                if gt_label:
                    output = model.forward_gt(x_curr, y_curr)
                else:
                    output = model(x_curr)
            acc += (output.max(1)[1] == y_curr).float().sum()

            all_outputs[counter * batch_size:(counter + 1) *
                    batch_size] = output.detach().cpu()

    assert (all_outputs==0).sum().item()==0

    #return raw 10000x100 outputs if return_raw is True
    if return_raw:
        return acc.item() / x.shape[0], all_outputs
    
    save_dir = f'./noadapt_stats/{corruption_type}'
    np.save(save_dir, all_outputs)
    return acc.item() / x.shape[0]

def save_fc_params(model, protocol:str):
    save_dir = './weights'
    for m in model.modules():
        if isinstance(m, nn.Linear):
            for name_p, p in m.named_parameters():
                if name_p in ['weight']:
                    np.save(save_dir+f'/{protocol}_weight',p.data.detach().cpu().numpy())
                elif name_p in ['bias']:
                    np.save(save_dir+f'/{protocol}_bias',p.data.detach().cpu().numpy())

def load_fc_params(model, protocol:str):
    weight_save_dir = './weights'+f'/{protocol}_weight.npy'
    bias_save_dir = './weights'+f'/{protocol}_bias.npy'
    weight = np.load(weight_save_dir)
    weight = torch.from_numpy(weight)
    bias = np.load(bias_save_dir)
    bias = torch.from_numpy(bias)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            for name_p, p in m.named_parameters():
                if name_p in ['weight']:
                    p.data = weight.cuda()
                elif name_p in ['bias']:
                    p.data = bias.cuda()

def check_freeze(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

def topk_accuracy(model: nn.Module,
                   x: torch.Tensor,
                   y: torch.Tensor,
                   batch_size: int = 100,
                   device: torch.device = None,
                   iteration = 1,
                   topk = 5):
    if device is None:
        device = x.device
    acc = np.zeros((topk))

    n_batches = math.ceil(x.shape[0] / batch_size)
    with torch.no_grad():
        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) *
                       batch_size].to(device)
            y_curr = y[counter * batch_size:(counter + 1) *
                       batch_size].to(device)

            for iter in range(iteration):
                output = model(x_curr)
            
            output_label = torch.argsort(output, axis=1).detach().cpu().numpy()
            output_label = output_label[:,::-1]
            for k in range(topk):
                acc[k] += (output_label[:,k] == y_curr.detach().cpu().numpy()).astype(float).sum()
        
        acc=np.cumsum(acc)

    return acc / x.shape[0]

def disable_batchnorm(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(False)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model

def split_train_val(x,y):
    x_train = x[:9000]
    y_train = y[:9000]
    x_test = x[9000:]
    y_test = y[9000:]

    return x_train, y_train, x_test, y_test