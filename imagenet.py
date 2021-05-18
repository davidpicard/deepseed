import sys

import torch
from torch import optim
import time
import argparse

from torch.utils.data import DataLoader

from datasets import *
from loss import *

batch_size = 256
v_batch_size = 50
epoch = 42

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", help="path to imagenet")
args = parser.parse_args()

train, val = build_imagenet(args.data_dir)
train_ds = DataLoader(train, batch_size=batch_size, num_workers=4, shuffle=True)
val_ds = DataLoader(val, batch_size=v_batch_size, num_workers=4)

# build model
model = torchvision.models.resnet50(pretrained=True)
model.to(device)


# optimization hparams
criterion = nn.CrossEntropyLoss()

# eval
model.eval()
val_loss = []
val_acc = []
for imgs, lbls in val_ds:
    imgs = imgs.to(device)
    lbls = lbls.to(device)
    outputs = model(imgs)
    val_loss.append(criterion(outputs, lbls))
    val_acc.append(((outputs.argmax(dim=1) == lbls).sum() / lbls.shape[0]).detach().cpu())
    print('val loss {:5.02f} val acc {:5.02f}'.format(torch.stack(val_loss).mean(), 100. * torch.stack(val_acc).mean()), end='\r')
print()






