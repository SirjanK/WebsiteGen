import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 32, 7, 1), nn.InstanceNorm2d(32), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, 3, 2), nn.InstanceNorm2d(64), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Conv2d(64, 128, 3, 2), nn.InstanceNorm2d(128), nn.ReLU())
        self.layer4 = [nn.Sequential(nn.Conv2d(128, 128, 3, 1), nn.Conv2d(128, 128, 3, 1)) for i in range(6)]
        self.layer5 = nn.Sequential(nn.ConvTranspose2d(128, 64, 3, 2), nn.InstanceNorm2d(64), nn.ReLU())
        self.layer6 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2), nn.InstanceNorm2d(32), nn.ReLU())
        self.layer7 = nn.Sequential(nn.Conv2d(32, 3, 7, 1), nn.InstanceNorm2d(3), nn.ReLU())

    def forward(self, x):
        x = nn.ReflectionPad2d((6, 5, 6, 5))(x)
        first = self.layer1(x)
        second = self.layer2(first)
        temp = self.layer3(second)
        pad = nn.ReflectionPad2d(2)
        for f in self.layer4:
            temp = pad(f(temp)) + temp
        fifth = self.layer5(temp)
        sixth = self.layer6(fifth)
        ret = self.layer7(sixth)
        ret = nn.ReflectionPad2d((2,1,2,1))(ret)
        return ret

"""
g = Generator()
a = np.zeros((1, 3, 128, 128)).astype(np.float32)
temp = torch.from_numpy(a)
print(temp.size())
(g(Variable(temp)))
"""