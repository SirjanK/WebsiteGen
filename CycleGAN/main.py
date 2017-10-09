from model import Model
import numpy as np
import torch
from torch.autograd import Variable
from imageParser import convert_image
import matplotlib.pyplot as plt
from random import shuffle


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
for i in range(10100, 10600):
    website_lst.append(convert_image(FILE_STRING_SITE + str(i) + '.png'))
for i in range(1000, 1300):
    mobile_lst.append(convert_image(FILE_STRING_MOBILE + str(i) + '.png'))
for i in range(2500, 2700):
    mobile_lst.append(convert_image(FILE_STRING_MOBILE + str(i) + '.png'))

shuffle(website_lst)
shuffle(mobile_lst)

#zero = np.zeros((1,3,128,128), dtype=np.float32)
#first = np.ones((1,3,128,128), dtype=np.float32)

#zero = convert_to_cuda_var(zero)
#first = convert_to_cuda_var(first)

# for i in range(NUM_ITER):
# if epoch > 100: # start decaying lr TODO

loss_val_web = []
loss_val_mob = []

for i in range(0, len(website_lst)):
    first = website_lst[i]
    model.train(first, 0)
    loss_val_web.append(model.calculate_loss(first, 0, None).data.numpy()[0])
    zero = mobile_lst[i]
    model.train(zero, 1)
    loss_val_mob.append(model.calculate_loss(zero, 1, None).data.numpy()[0])
    print(i)

loss_val_web = np.array(loss_val_web)
loss_val_mob = np.array(loss_val_mob)
np.savetxt('webloss.csv', loss_val_web)
np.savetxt('mobloss.csv', loss_val_mob)

i = 0
temp = website_lst[i].data.numpy()[0]
temp = np.transpose(temp, (2, 1, 0)).astype(np.float64)
print(temp.shape)
print(type(temp))

plt.figure()
#plt.subplot()
#plt.imshow(temp)
#plt.subplot()
transformed = model.evaluate(website_lst[i], 0).data.numpy()[0]
transformed = np.transpose(transformed, (2, 1, 0)).astype(np.float64)

plt.imshow(np.concatenate((temp, transformed), 1))
plt.show()


    #print(model.evaluate(first, 0))
