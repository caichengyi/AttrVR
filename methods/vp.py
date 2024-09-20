import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import Normalize

class PaddingVR(nn.Module):
    def __init__(self, out_size, input_size, init='zero', normalize=True):
        super(PaddingVR, self).__init__()
        mask = np.zeros((input_size, input_size))
        self.out_size = out_size
        if init == "zero":
            self.program = torch.nn.Parameter(data=torch.zeros(3, out_size, out_size))
        elif init == "randn":
            self.program = torch.nn.Parameter(data=torch.randn(3, out_size, out_size))
        self.normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        self.norm = normalize

        self.l_pad = int((out_size-input_size+1)/2)
        self.r_pad = int((out_size-input_size)/2)

        mask = np.repeat(np.expand_dims(mask, 0), repeats=3, axis=0)
        mask = torch.Tensor(mask)
        self.register_buffer("mask", F.pad(mask, (self.l_pad, self.r_pad, self.l_pad, self.r_pad), value=1))

    def forward(self, x):
        x = F.pad(x, (self.l_pad, self.r_pad, self.l_pad, self.r_pad), value=0) + torch.sigmoid(self.program) * self.mask
        if self.norm:
            x = self.normalize(x)
        return x


class WatermarkingVR(nn.Module):
    def __init__(self, size, pad):
        super(WatermarkingVR, self).__init__()

        self.size = size
        self.program = torch.nn.Parameter(data=torch.zeros(3, size, size))

        if size > 2*pad:
            mask = torch.zeros(3, size-2*pad, size-2*pad)
            self.register_buffer("mask", F.pad(mask, [pad for _ in range(4)], value=1))
        elif size == 2*pad:
            mask = torch.ones(3, size, size)
            self.register_buffer("mask", mask)

    def forward(self, x):
        x = x + self.program * self.mask
        return x
