import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable

LAST_LAYER_INPUT = 512*6*6


class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.layer1 = nn.Sequential(nn.Conv2d(3, 64, 4, 2), nn.LeakyReLU())
    self.layer2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2), nn.InstanceNorm2d(128), nn.LeakyReLU())
    self.layer3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2), nn.InstanceNorm2d(256), nn.LeakyReLU())
    self.layer4 = nn.Sequential(nn.Conv2d(256, 512, 4, 2), nn.InstanceNorm2d(512), nn.LeakyReLU())
    self.layer5 = nn.Linear(LAST_LAYER_INPUT, 1)
    self.layer6 = nn.Sigmoid()

  def forward(self, pictures):
    first = self.layer1(pictures)
    second = self.layer2(first)
    third = self.layer3(second)
    fourth = self.layer4(third)
    temp = fourth.view(-1, LAST_LAYER_INPUT)
    linear_output = self.layer5(temp)
    ret = self.layer6(linear_output)
    return ret


# Sanity Testing discriminator

d = Discriminator()
# picture = Image.open("random_sites/15000.png")
picture = Image.open("../random_sites/13.png").resize((128, 128))
data = np.asarray(picture, dtype=np.float32)
data = np.transpose(data, [2, 0, 1])
temp = torch.from_numpy(np.array([data]))
print(temp.size())
print(d.forward(Variable(temp)))
