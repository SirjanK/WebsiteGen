import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.relu = nn.LeakyReLU()
        self.layerOne = nn.Conv2d(3, 64, 4, 2)
        self.layerTwo = nn.Conv2d(64, 128, 4, 2)
        self.layerThree = nn.Conv2d(128, 256, 4, 2)
        self.layerFour = nn.Conv2d(256, 512, 4, 2)
        self.layerFive = nn.Linear(512*3*3, 1)

    def forward(self, picture):
        first = self.c(picture, self.layerOne, 0, True)
        second = self.c(first, self.layerTwo, 128)
        third = self.c(second, self.layerThree, 256)
        fourth = self.c(third, self.layerFour, 512)
        temp = fourth.view(-1, 512*3*3)
        ret = self.layerFive(temp)
        return ret


    def c(self, x, conv, k = 0, firstLayer = False):
        x = conv(x)
        if not firstLayer:
            x = nn.InstanceNorm2d(k)(x)
        x = self.relu(x)
        return x

d = Discriminator()
a = np.zeros((1, 3, 84, 84)).astype(np.float32)
temp = torch.from_numpy(a)

print(d.forward(Variable(temp)))