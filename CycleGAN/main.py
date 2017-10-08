from CycleGAN.model import Model
import numpy as np
import torch
from torch.autograd import Variable


USE_GPU = False
NUM_ITER = 1
NUM_EPOCHS = 200

epoch = 0


def convert_to_cuda_var(x):
  x = Variable(torch.from_numpy(x))
  if USE_GPU:
    x = x.cuda()
  return x

# TODO load images from X into memory
# TODO load images from Y into memory
# TODO remember to shuffle data
model = Model()
image_buffer = [] # stores the last 50 generated images

zero = np.zeros((1,3,128,128), dtype=np.float32)
first = np.ones((1,3,128,128), dtype=np.float32)

zero = convert_to_cuda_var(zero)
first = convert_to_cuda_var(first)


for i in range(NUM_ITER):
#  if epoch > 100: # start decaying lr TODO 

  model.train(first, 0)
  model.train(zero, 1)
  print(model.evaluate(first, 0))

