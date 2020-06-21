import os
import torch
import numpy as np 
import pandas as pd
import pretrainedmodels
import torch.nn as nn
from torch.nn import functional as F



class SEResNext50_32x4d(nn.Module):
    def __init__(self, pretrained="imagenet"):
        super().__init__()
        
        self.model = pretrainedmodels.__dict__["se_resnext50_32x4d"](pretrained=None)
        if pretrained is not None:
            self.model.load_state_dict(
                torch.load(
                    "input/se_resnext50_32x4d-a260b3a4.pth"
                )
            )
        self.out = nn.Linear(2048, 1)
    
    def forward(self, image, targets):
        batch, _, _, _ = image.shape
        x = self.model.features(image)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.reshape(batch, -1)
        out = self.out(x)
        loss = nn.BCEWithLogitsLoss()(out, targets.view(-1, 1).type_as(x))
        return out, loss
    

class SEResNeXt101_32x4d(nn.Module):
    def __init__(self, pretrained = "imagenet"):
        super().__init__()

        self.model = pretrainedmodels.__dict__['se_resnext101_32x4d'](pretrained = pretrained)
        #initialize target layers
        self.out = nn.Linear(2048,1) #grapheme root

    def forward(self, image, targets):
        batch, _, _, _ = image.shape
        x = self.model.features(image)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.reshape(batch, -1)
        out = self.out(x)
        loss = nn.BCEWithLogitsLoss()(out, targets.view(-1, 1).type_as(x))
        return out, loss
    

class NASnet(nn.Module):
    def __init__(self, pretrained = "imagenet+background"):
        super().__init__()

        self.model = pretrainedmodels.__dict__['nasnetalarge'](pretrained = pretrained)
        #initialize target layers
        self.out = nn.Linear(512,1) #grapheme root

    def forward(self, image, targets):
        batch, _, _, _ = image.shape
        x = self.model.features(image)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.reshape(batch, -1)
        out = self.out(x)
        loss = nn.BCEWithLogitsLoss()(out, targets.view(-1, 1).type_as(x))
        return out, loss