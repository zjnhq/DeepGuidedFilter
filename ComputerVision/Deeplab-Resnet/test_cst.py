import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pdb import *


class ConvSplit(nn.Module):
    def __init__(self, in_channels, out_channels= 1, kernel_size=3, stride=1, pad=1, dilation=1):
        super(ConvSplit, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, dilation=dilation, bias=True)
        self.conv.weights
        # self.bn = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        return F.sigmoid(self.conv(x), inplace=True)

class ConvSplitTree(nn.Module):
    def __init__(self, tree_depth, in_channels, out_channels= 1, kernel_size=3, stride=1, pad=1, dilation=1):
        super(ConvSplitTree, self).__init__()
        # self.conv.weights
        # self.bn = nn.BatchNorm2d(out_channels)
        if tree_depth>6:
            tree_depth = 6
        self.tree_depth = tree_depth
        self.convSplits = list()
        
        # for i in range(self.tree_depth):
        # self.convSplits.append(ConvSplit(in_planes, self.out_channels,kernel_size,stride,pad))
        self.convSplit = nn.Conv2d(1, self.tree_depth, kernel_size, stride=stride, padding=pad, dilation=dilation, bias=False).type(torch.FloatTensor)
        nn.init.normal_(self.convSplit.weight)
        self.numLeaves = int(2**self.tree_depth)
        self.out_channels = int(self.numLeaves * out_channels)
        self.convPred = nn.Conv2d(in_channels, self.out_channels, kernel_size, stride=stride, padding=pad, dilation=dilation, bias=True).type(torch.FloatTensor)
        nn.init.normal_(self.convPred.weight)
        self.normalize_weight_iter = True
        self.kernel_size = kernel_size

    def forward(self,x, data):
        # set_trace()
        if self.normalize_weight_iter:
            self.convSplit.weight.requires_grad = False
            self.convSplit.weight[self.convSplit.weight<0] *=0.0
            for i in range(self.tree_depth):
                normalizer = torch.sum(self.convSplit.weight[i])
                if normalizer< 0.1:
                    self.convSplit.weight[i] += 0.1 / self.kernel_size / self.kernel_size
                    normalizer =torch.sum(self.convSplit.weight[i]) 
                self.convSplit.weight[i] / normalizer
            self.convSplit.weight.requires_grad = True
        # splitWeight = torch.zeros(x.shape[0],self.numLeaves,x.shape[2],x.shape[3]) + 1./self.numLeaves
        # splitIndexStart = torch.zeros(x.shape[0],1, x.shape[2],x.shape[3]).type(torch.IntTensor)
        # splitIndexEnd = torch.zeros(x.shape[0],1, x.shape[2],x.shape[3]).type(torch.IntTensor) + self.numLeaves
        # return F.sigmoid(self.conv(x), inplace=True)
        value = F.sigmoid(self.convSplit(x))  
        leaf = torch.zeros(x.shape[0],1, x.shape[2],x.shape[3]).type(torch.IntTensor)
        for i in range(self.tree_depth):
            leaf = leaf * 2
            leaf += (value[:,i] < 0.6).view(x.shape[0],1, x.shape[2],x.shape[3])
        set_trace()
        value.view(x.shape[9])
        splitWeight = torch.zeros(x.shape[0],self.numLeaves,x.shape[2],x.shape[3]).type(torch.FloatTensor) + 1./self.numLeaves
        splitWeight[:,leaf] += 1
        # for i in range(self.tree_depth):
        #     splitIndexMedian = (splitIndexStart + splitIndexEnd)/2

        #     splitWeight[:,splitIndexStart:splitIndexMedian]= (2 - value[:,i,:,:]*2) * splitWeight[:,splitIndexStart:splitIndexMedian]
        #     splitWeight[:,splitIndexMedian:splitIndexEnd]= value[:,i,:,:] * 2 * splitWeight[:,splitIndexMedian:splitIndexEnd]
        #     split_dir = (value[:,i,:,:] < 0.5).type(torch.BoolTensor)
        #     splitIndexStart[:,split_dir] = splitIndexMedian[:,split_dir] 
        #     split_dir = 1 - split_dir
        #     splitIndexEnd[:,split_dir] = splitIndexMedian[:,split_dir]
        data = self.convPred(data)
        y = (data * splitWeight).sum(axis = 1)
        return y

class ConvSplitTree2(nn.Module):
    def __init__(self, tree_depth, in_channels, out_channels= 2, kernel_size=3, stride=1, pad=1, dilation=1, guide_in_channels =1):
        super(ConvSplitTree2, self).__init__()
        if tree_depth>6:
            tree_depth = 6
        self.tree_depth = tree_depth
        self.numLeaves = int(2**self.tree_depth)
        self.sum_out_channels = int(self.tree_depth * out_channels)
        self.convSplit = nn.Conv2d(guide_in_channels, self.sum_out_channels, kernel_size=1, stride=stride, padding=0, bias=False).type(torch.FloatTensor)
        nn.init.uniform_(self.convSplit.weight)
        self.out_channels = out_channels
        self.convPred = nn.Conv2d(in_channels, self.sum_out_channels, kernel_size, stride=stride, padding=pad, dilation=dilation, bias=True).type(torch.FloatTensor)
        nn.init.uniform_(self.convPred.weight)
        self.normalize_weight_iter = True
        self.kernel_size = kernel_size
        # self.maxpool = nn.MaxPool1d()

    def forward(self,x, data):
        if self.normalize_weight_iter:
            self.convSplit.weight.requires_grad = False
            self.convSplit.weight[self.convSplit.weight<0] *=0.0
            for i in range(self.tree_depth):
                normalizer = torch.sum(self.convSplit.weight[i])
                if normalizer< 0.1:
                    self.convSplit.weight[i] += 0.1 / self.kernel_size / self.kernel_size
                    normalizer =torch.sum(self.convSplit.weight[i]) 
                self.convSplit.weight[i] / normalizer
            self.convSplit.weight.requires_grad = True
        
        splitWeight =self.convSplit(x).view(x.shape[0],self.tree_depth,self.out_channels,x.shape[2],x.shape[3])
        splitWeight = F.softmax(splitWeight, dim=2)
        # .view(x.shape[0],self.sum_out_channels,x.shape[2],x.shape[3])
        data = self.convPred(data).view(x.shape[0],self.tree_depth,self.out_channels,x.shape[2],x.shape[3])
        data = data* splitWeight
        mask= splitWeight>0.5
        if torch.sum(mask) !=x.shape[0]*self.tree_depth*x.shape[2]*x.shape[3]:
            print("here some softmax value =0.5, adding some noise into it")
            splitWeight[:,0] +=0.001 
            splitWeight[:,1] -=0.001
            mask= splitWeight>0.5
        selected_data = torch.masked_select(data,mask).view(x.shape[0],self.tree_depth,x.shape[2],x.shape[3])
        y=torch.prod(selected_data,dim=1)
        # set_trace()
        # _, index = torch.max(splitWeight,dim = 2)
        # y= data[index].view(x.shape[0],self.tree_depth,x.shape[2],x.shape[3])
        
        # y = torch.pow(y,1.0/self.tree_depth)
        return y
        
        
import numpy as np
N = 20 
W = 30
H = 40
C = 8
depth = torch.tensor(np.random.rand(N,1,W,H)).float()
feat = torch.tensor(np.random.randn(N,C, W,H)).float()
cst = ConvSplitTree2(tree_depth = 3, in_channels= C).float()
y = cst(depth, feat)

# for it in range(10):
loss = torch.sum(y*y)
loss.backward()
set_trace()