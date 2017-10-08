import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable

def convertImage(imageName):
    picture = Image.open(imageName).resize((128, 128))
    pixels = np.asarray(picture)
    pixels = np.transpose(pixels, axes = (2, 0, 1)).astype(np.float32)
    temp = torch.from_numpy(np.array([pixels]))
    return Variable(temp)
