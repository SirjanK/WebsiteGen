from CycleGAN.model import Model
import numpy as np

NUM_ITER = 1000
NUM_EPOCHS = 200

epoch = 0


# load images A into memory
# load images B into memory
model = Model()
image_buffer = [] # stores the last 50 generated images

zero = np.zeros((1,3,128,128))
first = np.ones((1,3,128,128))

for i in range(NUM_ITER):
  if epoch > 100: # start decaying lr

  model.train(first, 0)
  model.train(zero, 1)

