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
        self.data = []
        self.upd = []
        self.mean = "null"

        self.dev = "null"

        self.covari = "null"

        for step, (x, y) in enumerate(train_loader):
            self.data.append(x)
        # for step, (x, y) in enumerate(train_loader):
        #     
        #     #y = y.detach().cpu().numpy()
        #     #print(output_x[0].shape,"Hello",y[0])
        #     if self.clasv == "null":
        #         self.clasv = output_x.detach().cpu().numpy()
        #     else:
        #         self.clasv = np.concatenate((self.clasv,output_x.detach().cpu().numpy()))
        #     # for i in range(len(y)):
            #     self.clasv.append(output_x[i].detach().cpu().numpy().flatten())
        # self.mean = temp.mean(axis=0)
        # self.dev = temp.std(axis=0)
        # self.covari = np.linalg.inv(np.cov(temp.transpose()))
        # print(self.mean.shape,self.dev.shape,self.covari.shape)

    def update(self,x):
        self.upd.append(x)

    def calc(self):
        self.upd+=self.data
        temp = "null"
        for x in self.upd:
            #print(type(x))
            output_x = self.model.ret_emb(x.to(self.args.device))
            if temp == "null":
                temp = output_x.detach().cpu().numpy()
            else:
                temp = np.concatenate((temp,output_x.detach().cpu().numpy()))
        self.mean = torch.tensor(temp.mean(axis=0),device=self.args.device)
        self.dev = torch.tensor(temp.std(axis=0),device=self.args.device)
        self.covari = torch.tensor(np.linalg.inv(np.cov(temp.transpose())),device=self.args.device,dtype=torch.float32)

    def clr(self):
        self.upd = []

    
    def mah_closer(self,x,lst=6):
        ans = 0
        lst = torch.tensor(1e20,device=self.args.device)
        #dist = spd.mahalanobis(x,self.mean,self.covari)
        #print(self.mean.shape,self.covari.shape,x.shape)
        #print(x.type(),self.mean.type(),self.covari.type())
        dist = torch.sqrt(((x-self.mean).unsqueeze(0)@self.covari@((x-self.mean).unsqueeze(0).transpose(1,0)))).float()
        #print("dist",dist)
        #print(((x-self.mean).unsqueeze(0)*self.covari).shape,dist.shape,((x-self.mean).unsqueeze(0).transpose(1,0)).shape,(x-self.mean).unsqueeze(0).shape)
        if (dist<lst).float()[0]:
            lst = dist.item()
            #print("lst",lst)
            ans = 1
        return lst,ans

    def batch_md(self,x,lst=6):
        x = self.model.ret_emb(x.to(self.args.device))
        #x = x.detach().cpu().numpy()
        ans = []
        for i in range(x.shape[0]):
            ans.append(self.mah_closer(x[i],lst)[0])
        return np.array(ans)


