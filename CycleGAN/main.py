from model import Model
import numpy as np
import torch
from torch.autograd import Variable
from imageParser import convert_image
import matplotlib.pyplot as plt


USE_GPU = False
NUM_ITER = 0  # edited idk what it was before rip
NUM_EPOCHS = 200
FILE_STRING_SITE = '../random_sites/'
FILE_STRING_MOBILE = '../random_sites_mobile/'

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
image_buffer = []  # stores the last 50 generated images

website_lst = []
mobile_lst = []
for i in range(10100, 10300):
    website_lst.append(convert_image(FILE_STRING_SITE + str(i) + '.png'))
    mobile_lst.append(convert_image(FILE_STRING_SITE + str(i) + '.png'))

#zero = np.zeros((1,3,128,128), dtype=np.float32)
#first = np.ones((1,3,128,128), dtype=np.float32)

#zero = convert_to_cuda_var(zero)
#first = convert_to_cuda_var(first)

# for i in range(NUM_ITER):
# if epoch > 100: # start decaying lr TODO

for i in range(0, len(website_lst)):
    first = website_lst[i]
    model.train(first, 0)
    zero = mobile_lst[i]
    model.train(zero, 1)

plt.figure()
plt.imshow(website_lst[0].data.numpy())
plt.figure()
plt.imshow(model.evaluate(website_lst[0], 0))

    #print(model.evaluate(first, 0))
