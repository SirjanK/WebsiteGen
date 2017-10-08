import torch
import torch.optim as optim
import torch.nn as nn
from CycleGAN.discriminator import Discriminator
from CycleGAN.generator import Generator
#from tensorboardX import SummaryWriter
#writer = SummaryWriter()

USE_GPU = False
RESUME = False
LAMBDA = 10

class Model:
  def __init__(self):
    self.d_X = Discriminator()
    self.d_Y = Discriminator()
    self.g_X = Generator()
    self.g_Y = Generator()
    self.optimizer_d_X = optim.Adam(self.d_X.parameters(), lr=0.0001)
    self.optimizer_d_Y = optim.Adam(self.d_Y.parameters(), lr=0.0001)
    self.optimizer_g_X = optim.Adam(self.g_X.parameters(), lr=0.0002)
    self.optimizer_g_Y = optim.Adam(self.g_Y.parameters(), lr=0.0002)
    self.L1Loss = nn.L1Loss()
    self.counter = 1
    if USE_GPU:
        self.d_X.cuda()
        self.d_Y.cuda()
        self.g_X.cuda()
        self.g_Y.cuda()
    if RESUME:
      # TODO Write this
      checkpoint = torch.load('checkpoint.tar')
      self.net.load_state_dict(checkpoint['state_dict'])
      self.optimizer.load_state_dict(checkpoint['optimizer'])



  def train(self, images, label):
    """
    :param x: input batch of images
    :param label: 0 for web (X), 1 for mobile (Y)
    :return:
    """
    images_mapped = self.evaluate(images, label)

    """ Calculate loss and apply gradient step """
    self.d_X.zero_grad()
    self.d_Y.zero_grad()
    self.g_X.zero_grad()
    self.g_Y.zero_grad()
    loss = self.calculate_loss(images, label, images_mapped)
    loss.backward()
    self.optimizer_d_X.step()
    self.optimizer_d_Y.step()
    self.optimizer_g_X.step()
    self.optimizer_g_Y.step()
    # writer.add_scalar("training_loss", loss.data.cpu().numpy(), self.counter)
    self.counter += 1

  def calculate_loss(self, images, label, images_mapped):
    # adversarial loss and cycle consistency loss
    if label == 0:
      adv_loss = self.d_Y(self.g_X(images)).norm(dim=1)
      cycle_loss = self.L1Loss(self.g_Y(self.g_X(images)), images)
      return adv_loss + LAMBDA * cycle_loss
    else:
      adv_loss = self.d_X(self.g_Y(images)).norm(dim=1)
      cycle_loss = self.L1Loss(self.g_X(self.g_Y(images)), images)
      return adv_loss + LAMBDA * cycle_loss


  def evaluate(self, images, label):
    if label == 0:
      return self.g_X(images)
    else:
      return self.g_Y(images)


