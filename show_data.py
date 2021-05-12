import torch
from torch import optim
import time

from models import *
from datasets import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


train_imgs, train_lbls, val_imgs, val_lbls = build_dataset(device=device)
n_train = len(train_lbls)
n_val = len(val_lbls)

augment = Augment()
augment.to(device).half()
x = train_imgs[0:5,...]
print(x.shape)
a = augment(x.to(device))
print(a.shape)
a -= a.min()
a /= torch.abs(a).max()
a = a.detach().cpu().permute(0, 2, 3, 1)
import matplotlib.pyplot as plt
for i in range(5):
  plt.subplot(1, 5, i+1)
  plt.imshow(a[i, ...])
plt.show()
