import numpy as np
import matplotlib.pyplot as plt

def plot_loss(label):
  """
  Plot the loss function for web if 0,
  mobile if 1.
  """
  if label == 0:
    loss = np.loadtxt('webloss.csv')
  else:
    loss = np.loadtxt('mobloss.csv')
  
  train_iter = np.r_[0:len(loss)]
  plt.plot(train_iter, loss)
  plt.show()
