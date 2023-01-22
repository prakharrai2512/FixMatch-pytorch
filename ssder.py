import argparse
import logging
import math
import os
import random
import shutil
import time
from collections import OrderedDict
import scipy.spatial.distance as spd
import numpy as np
import torch
from tqdm import tqdm

from dataset.cifar import DATASET_GETTERS
from utils import AverageMeter, accuracy

class SSDC:
    def __init__(self,model,train_loader,num_classes,args,class_labels=None,):
        self.args = args
        model.eval()
        self.model = model
        self.clasv = []
        for i in range(num_classes):
            self.clasv.append([])

        self.mean = []
        for i in range(num_classes):
            self.mean.append([])

        self.dev = []
        for i in range(num_classes):
            self.dev.append([])

        self.covari = []
        for i in range(num_classes):
            self.covari.append([])

        for step, (x, y) in enumerate(train_loader):
            output_x = self.model.ret_emb(x.to(self.args.device))
            y = y.detach().cpu().numpy()
            #print(output_x[0].shape,"Hello",y[0])
            for i in range(len(y)):
                self.clasv[y[i]].append(output_x[i].detach().cpu().numpy().flatten())
        
        for i in range(num_classes):
            temp = np.array(self.clasv[i])
            print(temp.shape)
            self.mean[i] = temp.mean(axis=0)
            self.dev[i] = temp.std(axis=0)
            self.covari[i] = np.linalg.inv(np.cov(temp.transpose()))
            print(self.mean[i].shape,self.dev[i].shape,self.covari[i].shape)


    def mah_close(self,x):
        x = self.model.ret_emb(x.to(self.args.device))
        x = x.detach().cpu().numpy()
        print(x.shape)
        lst = 1e20
        ans = 0
        for i in range(len(self.mean)):
            dist = spd.mahalanobis(x,self.mean[i],self.covari[i])
            if dist<lst:
                lst = dist
                ans = i
        return ans,lst
    
    def mah_closer(self,x):
        lst = 100000
        ans = 0
        for i in range(len(self.mean)):
            dist = spd.mahalanobis(x,self.mean[i],self.covari[i])
            if dist<lst:
                lst = dist
                ans = 1
                break
        return ans,lst

    def batch_md(self,x):
        x = self.model.ret_emb(x.to(self.args.device))
        x = x.detach().cpu().numpy()
        ans = []
        for i in range(x.shape[0]):
            ans.append(self.mah_closer(x[i])[0])
        return np.array(ans)


