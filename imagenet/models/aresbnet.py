'''AresBNet in PyTorch.

This AresBNet block is motivated from XNOR-Net and applied to ResNet below.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.autograd import Function

######################################################################################
class BinActive(torch.autograd.Function):
#{{{
    '''
    Binarize the input activations. 
    '''
    def forward(self, input):
        self.save_for_backward(input)
        size = input.size()
        input = input.add(1e-30).sign()
        return input

    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input
#}}}

class ShuffleBlock(nn.Module):
#{{{
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N,C,H,W = x.size()
        g = self.groups
        return x.view(N,g,C//g,H,W).permute(0,2,1,3,4).reshape(N,C,H,W)
#}}}

# apply batch normalization between subblocks
# ReLU after first convolution layer
# LeakyReLU slope is default
# 2x2 maxpooling in downsampling
# weight_decay = 0.00001
# epoch 400, batch 256: 
class BasicBlock_AresB(nn.Module): 
#{{{

    expansion = 2 
    def __init__(self, in_planes, planes, stride=1, suffle=False):
        super(BasicBlock_AresB, self).__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride
        self.suffle = suffle
        self.shuffle1 = ShuffleBlock(groups=2)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, groups=2)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.shuffle2 = ShuffleBlock(groups=2)
        self.conv2 = nn.Conv2d(2*planes, planes, kernel_size=3, stride=1, padding=1, bias=False, groups=2)
        self.maxpool2d= nn.MaxPool2d(3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(2*planes)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.bn3 = nn.BatchNorm2d(planes)
        
    def forward(self, x):
        if self.suffle:
          x = self.shuffle1(x)
        xa, xb = torch.chunk(x, 2, dim=1)
        x1 = BinActive()(x)
        x1 = self.conv1(x1)
        x1 = self.relu1(x1)
        x2 = self.bn1(x1)
        if self.stride != 1 :
          x3a = self.maxpool2d(x)
          x3b = x3a
        else:
          x3a = xa
          x3b = xb
        x3 = torch.cat((x2+x3a, x3b),1) 
        x3 = self.bn2(x3)
        x3 = self.shuffle2(x3)
        x4 = BinActive()(x3)
        x4 = self.conv2(x4)
        x4 = self.relu2(x4)
        x4 = self.bn3(x4)
        x5a, x5b = torch.chunk(x3, 2, dim=1)
        x5 = torch.cat((x4+x5a, x5b),1) 
        out = x5
        return out 
#}}}

## 3x3 maxpooling: grouped expanding AresB_9
## extending AresB_9x2 and bn
#class BasicBlock_AresB(nn.Module): 
##{{{
#    def __init__(self, in_planes, planes, stride=1, expansion=4, suffle=False):
#        super(BasicBlock_AresB, self).__init__()
#        self.in_planes = in_planes
#        self.planes = planes
#        self.stride = stride
#        self.expansion = expansion
#        self.suffle = suffle
#
#        self.shuffle1 = ShuffleBlock(groups=2)
#        self.conv1 = nn.Conv2d(in_planes*expansion, planes, kernel_size=3, stride=stride, padding=1, 
#                               bias=False, groups=expansion*2)
#        #self.relu1 = nn.PReLU(planes*expansion)
#        self.relu1 = nn.LeakyReLU(inplace=True)
#        self.bn1 = nn.BatchNorm2d(expansion*planes)
#        self.maxpool2d= nn.MaxPool2d(3, stride=2, padding=1)
#        self.bn2 = nn.BatchNorm2d(2*planes*expansion)
#        self.shuffle2 = ShuffleBlock(groups=2)
#        self.conv2 = nn.Conv2d(2*planes*expansion, planes, kernel_size=3, stride=1, 
#                               padding=1, bias=False, groups=expansion*2)
#        self.relu2 = nn.LeakyReLU(inplace=True)
#        self.bn3 = nn.BatchNorm2d(planes*self.expansion)
#        #self.relu2 = nn.PReLU(planes*expansion)
#        self.bn4 = nn.BatchNorm2d(2*planes*self.expansion)
#        
#    def forward(self, x):
#        if self.suffle:
#          x = self.shuffle1(x)
#        xa, xb = torch.chunk(x, 2, dim=1)
#        x1 = BinActive()(x)
#        x1 = self.conv1(x1)
#        #x1 = x1.unsqueeze(1)
#        #x1 = x1.expand(-1, self.expansion, -1, -1, -1)
#        #x1 = x1.reshape(x1.shape[0], -1, x1.shape[3], x1.shape[4])
#        #x1 = x1.reshape(x1.shape[0], -1, x1.shape[3], x1.shape[4])
#        #x1 = x1.repeat(1, self.expansion, 1, 1)
#        x1 = self.relu1(x1)
#        x2 = self.bn1(x1)
#        if self.stride != 1 :
#          x3a = self.maxpool2d(x)
#          x3b = x3a
#        else:
#          x3a = xa
#          x3b = xb
#        x3 = torch.cat((x2+x3a, x3b),1) 
#        x3 = self.bn2(x3)
#        x3 = self.shuffle2(x3)
#        x4 = BinActive()(x3)
#        x4 = self.conv2(x4)
#        #x4 = x4.unsqueeze(1)
#        #x4 = x4.expand(-1, self.expansion, -1, -1, -1)
#        #x4 = x4.reshape(x4.shape[0], -1, x4.shape[3], x4.shape[4])
#        #x4 = x4.repeat(1, self.expansion, 1, 1)
#        x4 = self.relu2(x4)
#        x4 = self.bn3(x4)
#        x5a, x5b = torch.chunk(x3, 2, dim=1)
#        x5 = torch.cat((x4+x5a, x5b),1) 
#        #x5 = self.bn4(x5)
#        out = x5
#        return out 
##}}}

class AresBNet(nn.Module):
#{{{
    def __init__(self, block, num_blocks, num_classes=1000):
        super(AresBNet, self).__init__()
        self.in_planes = 64 * 2

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64*2)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, suffle=False)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, suffle=True)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, suffle=True)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, suffle=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))            
        #self.dropout = nn.Dropout()
        self.fc = nn.Linear(512, num_classes)    

    def _make_layer(self, block, planes, num_blocks, stride, suffle):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        if suffle:
          dosuffle=True
        else:  
          dosuffle=False
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, dosuffle))
            self.in_planes = planes * block.expansion
            dosuffle = True
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.leaky_relu(out)
        out = self.maxpool(out)
        out = torch.cat(2*[out], 1)
        out = self.bn2(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.chunk(out, 2, dim=1)[0]
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)          
        return out
#}}}


def AresBNet10():
    return AresBNet(BasicBlock_AresB, [1,1,1,1])

def AresBNet18():
    return AresBNet(BasicBlock_AresB, [2,2,2,2])

def AresBNet34(): 
    return AresBNet(BasicBlock_AresB, [3,4,6,3])
