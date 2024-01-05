import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math


class ArcMarginProduct(nn.Module):
    def __init__(self,
                 feature_dim,
                 class_dim,
                 margin=0.5,
                 scale=64.0,):
        super().__init__()
        self.feature_dim = feature_dim
        self.class_dim = class_dim
        self.margin = margin
        self.scale = scale
        self.weight = Parameter(torch.FloatTensor(feature_dim, class_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(self.margin) * (-1)
        self.mm = math.sin(margin) * margin

    def forward(self, input, label):
        input_norm = torch.sqrt(torch.sum(torch.square(input), dim=1, keepdim=True))
        input = torch.divide(input, input_norm)

        weight_norm = torch.sqrt(torch.sum(torch.square(self.weight), dim=0, keepdim=True))
        weight = torch.divide(self.weight, weight_norm)
        cos = torch.matmul(input, weight)

        sin = torch.sqrt(1.0 - torch.square(cos) + 1e-6)
        phi = cos * self.cos_m - sin * self.sin_m
        mask = (cos > self.threshold).float()
        phi = torch.multiply(mask, phi) + torch.multiply((1.0 - mask), cos - self.mm)

        one_hot = torch.nn.functional.one_hot(label, self.class_dim)
        one_hot = torch.squeeze(one_hot, dim=1)
        output = torch.multiply(one_hot, phi) + torch.multiply((1.0 - one_hot), cos)
        output = output * self.scale
        return output