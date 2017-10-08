import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable


class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.layer1 = nn.Sequential(nn.Conv2d(3, 64, 4, 2), nn.LeakyReLU())
    self.layer2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2), nn.InstanceNorm2d(128), nn.LeakyReLU())
    self.layer3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2), nn.InstanceNorm2d(256), nn.LeakyReLU())
    self.layer4 = nn.Sequential(nn.Conv2d(256, 512, 4, 2), nn.InstanceNorm2d(512), nn.LeakyReLU())
    self.layer5 = nn.Linear(512*3*3, 1)

  def forward(self, pictures):
    first = self.layer1(pictures)
    second = self.layer2(first)
    third = self.layer3(second)
    fourth = self.layer4(third)
    temp = fourth.view(-1, 512*3*3)
    ret = self.layer5(temp)
    return ret



d = Discriminator()
'''
picture = Image.open("random_sites/15000.png")
data = np.asarray(picture)
temp = torch.from_numpy(np.array([data]))
print(temp.size())
print(d.forward(Variable(temp)))
'''

#a = np.zeros((1, 3, 84, 84)).astype(np.float32)
#temp = torch.from_numpy(a)
#print(d.forward(Variable(temp)))