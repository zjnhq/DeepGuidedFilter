import torch
import torch.nn as nn

from torch.nn import functional as F
from guided_filter_pytorch.guided_filter import GuidedFilter
from pdb import set_trace
AFFINE = True

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=AFFINE)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=AFFINE)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation_=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=AFFINE)
        for i in self.bn1.parameters():
            i.requires_grad = False
        padding = 1
        if dilation_ == 2:
            padding = 2
        elif dilation_ == 4:
            padding = 4
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=padding, bias=False, dilation=dilation_)
        self.bn2 = nn.BatchNorm2d(planes, affine=AFFINE)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=AFFINE)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Classifier_Module(nn.Module):
    def __init__(self, dilation_series, padding_series, NoLabels):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(2048, NoLabels, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, NoLabels, need_mid_features= False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=AFFINE)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation__=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation__=4)
        self.layer5 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], NoLabels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.need_mid_features = need_mid_features

    def _make_layer(self, block, planes, blocks, stride=1, dilation__=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation__ == 2 or dilation__ == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=AFFINE),
            )
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation_=dilation__, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation_=dilation__))

        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels):
        return block(dilation_series, padding_series, NoLabels)
    
    def forward(self, x):
        if self.need_mid_features:
            return self.forward_complex(x)
        else:
            return self.forward_simple(x)

    def forward_complex(self, x):
        # set_trace()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        low_feat = x
        x2 = self.maxpool(x)
        x2 = self.layer1(x2)
        x2 = self.layer2(x2)
        mid_feat = x2
        x3 = self.layer3(x2)
        x3 = self.layer4(x3)
        x3 = self.layer5(x3)

        return x3, low_feat, mid_feat

    def forward_simple(self, x):
        # set_trace()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        return x


class MS_Deeplab(nn.Module):
    def __init__(self, block, NoLabels, dgf, dgf_r, dgf_eps):
        super(MS_Deeplab, self).__init__()
        self.Scale = ResNet(block, [3, 4, 23, 3], NoLabels, need_mid_features = True)  # changed to fix #4

        # DGF
        self.dgf = dgf

        if self.dgf:
            self.guided_map_conv1 = nn.Conv2d(3, 64, 1)
            self.guided_map_relu1 = nn.ReLU(inplace=True)
            self.guided_map_conv2 = nn.Conv2d(64, NoLabels, 1)

            self.guided_filter = GuidedFilter(dgf_r, dgf_eps)

    def forward(self, x, im=None):
        output, low_feat, mid_feat = self.Scale(x)

        if self.dgf:
            g = self.guided_map_relu1(self.guided_map_conv1(im))
            g = self.guided_map_conv2(g)

            output = F.interpolate(output, im.size()[2:], mode='bilinear', align_corners=True)

            output = self.guided_filter(g, output)

        return output

class ConvSplitTree2(nn.Module):
    def __init__(self, tree_depth, in_channels, out_channels= 2, kernel_size=3, stride=1, pad=1, dilation=1, guide_in_channels =1, n_split=2, resize_small=1, is_regression=1):
        super(ConvSplitTree2, self).__init__()
        if tree_depth>6:
            tree_depth = 6
        self.tree_depth = tree_depth 
        self.is_regression = is_regression
        if self.is_regression ==1:
            self.n_split = n_split
        else:
            self.n_split = 1
        self.tree_nodes = (self.tree_depth * self.n_split)
        self.sum_out_channels = int(self.tree_nodes * out_channels)
        if self.is_regression ==1:
            self.convSplit = nn.Conv2d(guide_in_channels, self.sum_out_channels, kernel_size=1, stride=stride, padding=0, bias=False).type(torch.FloatTensor)
            nn.init.uniform_(self.convSplit.weight)
        self.out_channels = out_channels
        self.convPred = nn.Conv2d(in_channels, self.sum_out_channels, kernel_size, stride=stride, padding=pad, dilation=dilation, bias=True).type(torch.FloatTensor)
        nn.init.uniform_(self.convPred.weight)
        self.normalize_weight_iter = False
        self.kernel_size = kernel_size
        self.resize_small = resize_small


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
        if self.is_regression:
            return self.forward_regression(x,data)
        else:
            return self.forward_classification(x,data)
    def forward_regression(self,x, data): # x is features
        # set_trace()
        score =self.convSplit(x).view(x.shape[0],self.tree_depth,self.n_split, self.out_channels,x.shape[2],x.shape[3])
        data = self.convPred(data).view(x.shape[0],self.tree_depth,self.n_split,self.out_channels,x.shape[2],x.shape[3])
        score = score *0.1
        score = F.softmax(score, dim=2)
        data = torch.sum(torch.sum(score * data, dim=2),dim=1)
        final_score,_ = torch.max(score, dim=2)
        final_score = torch.prod(final_score, dim=1)
        y= final_score * data
        return y

    def forward_classification(self,x, data): # x is probability vector for each class, channels of x should equal to output channels
        # set_trace()
        score = x.view(x.shape[0],self.tree_depth,self.out_channels,x.shape[2],x.shape[3])
        data = self.convPred(data).view(x.shape[0],self.tree_depth,self.out_channels,x.shape[2],x.shape[3])
        score = F.softmax(score, dim=2)
        data = torch.sum(score * data, dim=1)
        y = data
        # final_score,_ = torch.sum(score, dim=1)
        # # final_score = torch.sum(final_score, dim=1) 
        # y= final_score * data
        return y

# class CST_Deeplab(nn.Module):
#     def __init__(self, block, NoLabels, dgf):
#         super(CST_Deeplab, self).__init__()
#         self.Scale = ResNet(block, [3, 4, 23, 3], NoLabels, need_mid_features= True)  # changed to fix #4

#         # DGF
#         self.dgf = dgf

#         if self.dgf:
#             guide_in_channels = 16
#             self.guided_map_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3)
#             self.guided_map_relu1 = nn.ReLU(inplace=True)
#             self.guided_map_conv2 = nn.Conv2d(64, guide_in_channels, kernel_size=1)
#             self.guided_map_relu2 = nn.ReLU(inplace=True)
 
#             # self.cst = ConvSplitTree2(3, NoLabels, out_channels=NoLabels, guide_in_channels = guide_in_channels)
#             self.cst2 = ConvSplitTree2(tree_depth=3, in_channels=64, out_channels= NoLabels, kernel_size=1, stride=1, pad=0, dilation=1, guide_in_channels =guide_in_channels, n_split=2,resize_small=1)
#             self.cst3 = ConvSplitTree2(tree_depth=3, in_channels=512, out_channels= NoLabels, kernel_size=1, stride=1, pad=0, dilation=1, guide_in_channels =guide_in_channels, n_split=2, resize_small =1)

#     def forward(self, x, im=None):
#         output, low_feat, mid_feat = self.Scale(x)

#         if self.dgf:
#             g = self.guided_map_relu1(self.guided_map_conv1(im))
#             g = self.guided_map_relu2(self.guided_map_conv2(g))
#             output = F.interpolate(output, im.size()[2:], mode='bilinear', align_corners=True)

#             output2 = self.cst2(g, low_feat)
#             output3 = self.cst3(g, mid_feat)
#             output2 = F.interpolate(output2, im.size()[2:], mode='bilinear', align_corners=True)
#             output3 = F.interpolate(output3, im.size()[2:], mode='bilinear', align_corners=True)
#             output += (output2 + output3)*0.5
#         return output

class CST_Deeplab2(nn.Module):
    def __init__(self, block, NoLabels, dgf, dgf_r, dgf_eps):
        super(CST_Deeplab2, self).__init__()
        self.Scale = ResNet(block, [3, 4, 23, 3], NoLabels, need_mid_features= True)  # changed to fix #4

        # DGF
        self.dgf = dgf

        if self.dgf:
            feature_in_channels = 16
            guide_in_channels = 32
            self.guided_map_conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=3)
            self.guided_map_relu1 = nn.ReLU(inplace=True)
            self.guided_map_conv2 = nn.Conv2d(64, feature_in_channels, kernel_size=1)
            self.guided_map_relu2 = nn.ReLU(inplace=True)
            self.guided_map_conv3 = nn.Conv2d(512, feature_in_channels, kernel_size=1)
            self.guided_map_relu3 = nn.ReLU(inplace=True)

            self.guided_map_conv5= nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
 
            # self.cst = ConvSplitTree2(3, NoLabels, out_channels=NoLabels, guide_in_channels = guide_in_channels)
            self.cst2 = ConvSplitTree2(tree_depth=3, in_channels= feature_in_channels, out_channels= NoLabels, kernel_size=1, stride=1, pad=0, dilation=1, guide_in_channels =guide_in_channels, n_split=2,resize_small=0)
            self.cst3 = ConvSplitTree2(tree_depth=3, in_channels= feature_in_channels, out_channels= NoLabels, kernel_size=1, stride=1, pad=0, dilation=1, guide_in_channels =guide_in_channels, n_split=2,resize_small=0)
            self.cst4 = ConvSplitTree2(tree_depth=3, in_channels= feature_in_channels, out_channels= NoLabels, kernel_size=1, stride=1, pad=0, dilation=1, guide_in_channels =NoLabels, n_split=2,resize_small=0, is_regression=0)
            self.cst5 = ConvSplitTree2(tree_depth=3, in_channels= feature_in_channels, out_channels= NoLabels, kernel_size=1, stride=1, pad=0, dilation=1, guide_in_channels =NoLabels, n_split=2,resize_small=0, is_regression=0)

            self.guided_filter = GuidedFilter(dgf_r, dgf_eps)

    def forward(self, x, im=None):
        output, low_feat, mid_feat = self.Scale(x)

        if self.dgf:
            g = self.guided_map_relu1(self.guided_map_conv1(im))
            low_feat = self.guided_map_relu2(self.guided_map_conv2(low_feat))
            mid_feat = self.guided_map_relu3(self.guided_map_conv3(mid_feat))
            output = F.interpolate(output, im.size()[2:], mode='bilinear', align_corners=True)

            output2 = self.cst2(g, low_feat)
            output3 = self.cst3(g, mid_feat)

            output2 = F.interpolate(output2, im.size()[2:], mode='bilinear', align_corners=True)
            output3 = F.interpolate(output3, im.size()[2:], mode='bilinear', align_corners=True)
            cat_output = torch.cat((output,output2,output3),1)
            # set_trace()
            output4 = self.cst4(cat_output, low_feat)
            output5 = self.cst5(cat_output, mid_feat)
            output4 = F.interpolate(output4, im.size()[2:], mode='bilinear', align_corners=True)
            output5 = F.interpolate(output5, im.size()[2:], mode='bilinear', align_corners=True)
            # set_trace()
            output += (output2 + output3)*0.15
            output += (output4 + output5)*0.15
            g_c1 = self.guided_map_conv5(g)

            output = self.guided_filter(g_c1, output)
        return output

def Res_Deeplab(NoLabels=21, dgf=False, dgf_r=4, dgf_eps=1e-2):
    model = MS_Deeplab(Bottleneck, NoLabels, dgf, dgf_r, dgf_eps)
    return model
def Cst_Deeplab(NoLabels=21, dgf=True, dgf_r=4, dgf_eps=1e-2):
    model = CST_Deeplab2(Bottleneck, NoLabels, dgf=dgf, dgf_r=dgf_r, dgf_eps=dgf_eps)
    return model