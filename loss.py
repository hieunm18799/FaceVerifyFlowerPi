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

    def forward(self, inputs, labels):
        inputs_norm = torch.sqrt(torch.sum(torch.square(inputs), dim=1, keepdim=True))
        inputs = torch.divide(inputs, inputs_norm)

        weight_norm = torch.sqrt(torch.sum(torch.square(self.weight), dim=0, keepdim=True))
        weight = torch.divide(self.weight, weight_norm)
        cos = torch.matmul(inputs, weight)

        for index, label in enumerate(labels):
            cos[index][label] = cos[index][label] *self.cos_m - self.sin_m * math.sqrt(1.0 - cos[index][label]**2)

        return cos * self.scale