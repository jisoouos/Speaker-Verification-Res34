from model import ResNet
from ResBlock import BasicBlock
import torch.nn as nn
import torch
from newmodel import resnet18

class ResNet18(nn.Module):
    def __init__(self):
            super(ResNet18, self).__init__()
            #self.aa = ResNet(BasicBlock,[2,2,2,2])
            self.aa = resnet18(channel_size=1,inplane=64,embedding_size=512)

    def forward(self, input_tensor: torch.Tensor):
            return self.aa(input_tensor)