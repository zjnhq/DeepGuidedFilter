
from __future__ import print_function
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
    def __init__(self, tree_depth, in_channels, out_channels= 2, kernel_size=3, stride=1, pad=1, dilation=1, guide_in_channels =1, n_split=2):
        super(ConvSplitTree2, self).__init__()
        if tree_depth>6:
            tree_depth = 6
        self.tree_depth = tree_depth 
        self.n_split = n_split
        self.tree_nodes = (tree_depth * n_split)
        # self.numLeaves = int(2**self.tree_depth)
        self.sum_out_channels = int(self.tree_nodes * out_channels)
        self.convSplit = nn.Conv2d(guide_in_channels, self.sum_out_channels, kernel_size=1, stride=stride, padding=0, bias=False).type(torch.FloatTensor)
        nn.init.uniform_(self.convSplit.weight)
        self.out_channels = out_channels
        self.convPred = nn.Conv2d(in_channels, self.sum_out_channels, kernel_size, stride=stride, padding=pad, dilation=dilation, bias=True).type(torch.FloatTensor)
        nn.init.uniform_(self.convPred.weight)
        self.normalize_weight_iter = False
        self.kernel_size = kernel_size
        # self.maxpool = nn.MaxPool1d()
        self.resize_small = 1
        self.softmax_weight = 2.0

    def forward_obsolete(self,x, data):
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
        
        score =self.convSplit(x).view(x.shape[0],self.tree_depth,self.out_channels,x.shape[2],x.shape[3])
        # score = F.softmax(score, dim=1)
        score = F.sigmoid(score)
        splitWeight = torch.prod(splitWeight, dim=1)
        # .view(x.shape[0],self.sum_out_channels,x.shape[2],x.shape[3])
        data = self.convPred(data).view(x.shape[0],self.tree_depth,self.out_channels,x.shape[2],x.shape[3])
        # set_trace()
        maxweight, index = torch.max(splitWeight, dim=1)
        index = index.view(x.shape[0],1,self.out_channels*x.shape[2]*x.shape[3]).transpose(1,2).view(-1,1)
        data = data* splitWeight
        data=data.view(x.shape[0],self.tree_depth,self.out_channels*x.shape[2]*x.shape[3]).transpose(1,2)#.view(-1, self.tree_depth)
        self.n_pixel_out= x.shape[0]*self.out_channels*x.shape[2]*x.shape[3]
        self.range_index = torch.Tensor(np.arange(int(self.n_pixel_out))).type(torch.LongTensor)
        selected_data = data.reshape(self.n_pixel_out, self.tree_depth)[self.range_index, index.view(-1)]
        selected_data = selected_data.view(x.shape[0], self.out_channels, x.shape[2], x.shape[3])

        # selected_data= torch.index_select(data,2,index.view(x.shape[0]*self.out_channels*x.shape[2]*x.shape[3]))
        # mask= splitWeight==maxweight
        # if torch.sum(mask) !=x.shape[0]*self.out_channels*x.shape[2]*x.shape[3]:
        #     print("here some softmax value =0.5, adding some noise into it")
        #     splitWeight[:,0] +=0.001 
        #     splitWeight[:,1] -=0.001
        #     mask= splitWeight>0.5
        # set_trace()
        # selected_data = torch.masked_select(data,mask).view(x.shape[0],self.tree_depth,self.out_channels,x.shape[2],x.shape[3])
        y=torch.prod(selected_data,dim=1)
        # set_trace()
        # _, index = torch.max(splitWeight,dim = 2)
        # y= data[index].view(x.shape[0],self.tree_depth,x.shape[2],x.shape[3])
        
        # y = torch.pow(y,1.0/self.tree_depth)
        return y

    def set_eval(self, eval_= True):
        if eval_:
            self.softmax_weight = 1.0
        else:
            self.softmax_weight = 0.2

    def forward(self,x, data):
        if x.shape[2]< data.shape[2]:
            if self.resize_small ==1:
                data = F.interpolate(data, x.size()[2:], mode='bilinear', align_corners=True)
            else:
                x = F.interpolate(x, data.size()[2:], mode='bilinear', align_corners=True)

        if x.shape[2]> data.shape[2]:
            if self.resize_small ==1:
                x = F.interpolate(x, data.size()[2:], mode='bilinear', align_corners=True)
            else:
                data = F.interpolate(data, x.size()[2:], mode='bilinear', align_corners=True)
        # set_trace()
        score =self.convSplit(x).view(x.shape[0],self.tree_depth,self.n_split, self.out_channels,x.shape[2],x.shape[3])
        data = self.convPred(data).view(x.shape[0],self.tree_depth,self.n_split,self.out_channels,x.shape[2],x.shape[3])
        score = F.softmax(score, dim=2)
        score = score * self.softmax_weight
        data = torch.sum(torch.sum(score * data, dim=2),dim=1)
        # score = F.sigmoid(score)
        final_score,_ = torch.max(score, dim=2)
        final_score = torch.prod(final_score, dim=1)
        y= final_score * data
        return y
        
        
import numpy as np
def test_init():
    N = 20 
    W = 30
    H = 40
    C = 8
    CG = 4
    depth = torch.tensor(np.random.rand(N,CG,W,H)).float()
    feat = torch.tensor(np.random.randn(N,C, W,H)).float()
    cst = ConvSplitTree2(tree_depth = 3, in_channels= C, out_channels=10, guide_in_channels = CG).float()
    y = cst(depth, feat)

    # for it in range(10):
    loss = torch.sum(y*y)
    loss.backward()
    set_trace()


import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class CST_Net(nn.Module):
    def __init__(self):
        super(CST_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 32, 3, 1)
        self.conv4 = nn.Conv2d(32, 64, 3, 1)
        # self.dropout1 = nn.Dropout(0.25)
        # self.dropout2 = nn.Dropout(0.5)
        self.cst = ConvSplitTree2(tree_depth=3, in_channels=64, out_channels= 10, guide_in_channels = 32)
        self.cst2 = ConvSplitTree2(tree_depth=3, in_channels=32, out_channels= 10, guide_in_channels = 64)
        # self.fc1 = nn.Linear(9216, 128)
        # self.fc2 = nn.Linear(128, 10)

    def set_eval(self, eval_):
        self.cst.set_eval(eval_)
        self.cst2.set_eval(eval_)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x2 = F.max_pool2d(x, 2)
        x2 = self.conv3(x2)
        x2 = F.relu(x2)
        x2 = self.conv4(x2)
        x2 = F.relu(x2)
        y = self.cst(x,x2)
        y += self.cst2(x2,x)*0.5
        y= torch.mean(torch.mean(y, dim = 3),dim=2)

        output = F.log_softmax(y, dim=1)
        return output

def get_1x_lr_params_NOscale(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = []
    b.append(model.conv1)
    b.append(model.conv2)
    b.append(model.conv3)
    b.append(model.conv4)

    for i in range(len(b)):
        for j in b[i].modules():
            for k in j.parameters():
                if k.requires_grad:
                    yield k


def get_10x_lr_params(args, model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = []
    b.append(model.cst.parameters())
    b.append(model.cst2.parameters())


    for j in range(len(b)):
        for i in b[j]:
            yield i


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 32, 3, 1)
        self.conv4 = nn.Conv2d(32, 64, 3, 1)
        self.conv5 = nn.Conv2d(64, 10, 3, 1)
        # self.cst = ConvSplitTree2(tree_depth=3, in_channels=64, out_channels= 10, guide_in_channels = 32)
        # self.cst2 = ConvSplitTree2(tree_depth=3, in_channels=32, out_channels= 10, guide_in_channels = 64)
        # self.fc1 = nn.Linear(9216, 128)
        # self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x2 = F.max_pool2d(x, 2)
        x2 = self.conv3(x2)
        x2 = F.relu(x2)
        x2 = self.conv4(x2)
        x2 = F.relu(x2)
        y = self.conv5(x2)
        y= torch.mean(torch.mean(y, dim = 3),dim=2)

        output = F.log_softmax(y, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        # set_trace()
        loss.backward()
        model.cst.convSplit.weight.grad = model.cst.convSplit.weight.grad*10.0
        model.cst2.convSplit.weight.grad = model.cst2.convSplit.weight.grad*10.0
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def test_attack(model, device, test_loader):
    # model.eval()
    test_loss = 0
    test_loss_attack = 0
    correct = 0
    correct_attack = 0
    use_attack_ = True
    attack_maxstepsize = 0.1
    # with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_loader):
        # if batch_idx %10==0:print(batch_idx)
        data, target = data.to(device), target.to(device)
        output = model(data) 
        test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        if use_attack_:
            data_original = data.clone().detach()
            data.requires_grad = True
            data.retain_grad()
            output = model(data)   

            input_upper_limit= (data_original + attack_maxstepsize).detach()
            input_lower_limit = (data_original - attack_maxstepsize).detach()
            steps = min(int(attack_maxstepsize/0.001),8)
            attack_stepsize = attack_maxstepsize/steps
            # attack_stepsize = 100000.0
            
            for attack_iter in range(steps):
                loss = F.nll_loss(output, target)
                loss.backward()
                # input.abs().mean()/(input.grad.abs().mean())
                data.detach()
                data.data = data.data + data.grad.sign() * attack_stepsize
                data.data[data.data>input_upper_limit] = input_upper_limit.data[data.data>input_upper_limit]
                data.data[data.data<input_lower_limit] = input_lower_limit.data[data.data<input_lower_limit]
                data.requires_grad= True
                data.retain_grad()
                output = model(data)    

            del data_original, input_upper_limit, input_lower_limit
            test_loss_attack += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct_attack += pred.eq(target.view_as(pred)).sum().item()
            del data, output, target

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    if use_attack_:
        test_loss_attack /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss_attack, correct_attack, len(test_loader.dataset),
            100. * correct_attack / len(test_loader.dataset)))



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=40, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.3, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=500, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    print("CST_Net:")
    model = CST_Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    args.epochs = 4

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        model.set_eval(False)
        train(args, model, device, train_loader, optimizer, epoch)
        model.set_eval(True)
        test_attack(model, device, test_loader)
        scheduler.step()
        # break
    del model, optimizer, scheduler

    print("ConvNet:")
    model = ConvNet().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test_attack(model, device, test_loader)
        scheduler.step()
        # break

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()