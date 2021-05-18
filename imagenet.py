import sys

import torch
from torch import optim
import time
import argparse

from torch.utils.data import DataLoader

from datasets import *
from loss import *

batch_size = 512
batch_size_ft = 64
v_batch_size = 50
epoch = 2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


def eval(model):
    model.eval()
    val_loss = []
    val_acc = []
    i = 1
    for imgs, lbls in val_ds:
        imgs = imgs.to(device)
        lbls = lbls.to(device)
        outputs = model(imgs)
        val_loss.append(criterion(outputs, lbls).detach().cpu())
        val_acc.append(((outputs.argmax(dim=1) == lbls).sum() / lbls.shape[0]).detach().cpu())
        print('{}/{} val loss {:5.02f} val acc {:5.02f}'.format(i, n_val, torch.stack(val_loss).mean(), 100. * torch.stack(val_acc).mean()), end='\r')
        i += 1
    print()
    return torch.stack(val_loss).mean(), 100. * torch.stack(val_acc).mean()


#############


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", help="path to imagenet")
parser.add_argument("--eval_pretrained", type=bool, default=False)
args = parser.parse_args()

train, val = build_imagenet(args.data_dir)
train_ds = DataLoader(train, batch_size=batch_size, num_workers=10, shuffle=True)
val_ds = DataLoader(val, batch_size=v_batch_size, num_workers=10)
n_train = len(train_ds)
n_val = len(val_ds)

# build model
model = torchvision.models.resnet50(pretrained=True)
if not args.eval_pretrained:
    torch.nn.init.kaiming_uniform_(model.fc.weight)
    torch.nn.init.normal_(model.fc.bias)

for p in model.parameters():
    p.requires_grad = False
if not args.eval_pretrained:
    model.fc.weight.requires_grad = True
    model.fc.bias.requires_grad = True

model.to(device)


# optimization hparams
criterion = nn.CrossEntropyLoss()

if not args.eval_pretrained:
    criterion2 = CrossEntropyLabelSmooth(num_classes=10, epsilon=0.3)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True, weight_decay=0.0001)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, eta_min=0.001)

    # training loop
    print('Training last layer')
    t_start = time.time()
    for e in range(epoch):  # loop over the dataset multiple times
        running_loss = []
        running_acc = []
        start = time.time()
        i = 1
        for imgs, lbls in train_ds:
            imgs = imgs.to(device)
            lbls = lbls.to(device)

            optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss2 = criterion2(outputs, lbls)
            loss = loss + loss2

            loss.backward()
            optimizer.step()
            # print statistics
            running_loss.append(loss.detach().cpu())
            running_acc.append(((outputs.argmax(dim=1) == lbls).sum() / lbls.shape[0]).detach().cpu())

            print('{}/{} loss: {:5.02f} acc: {:5.02f} in {:6.01f}'.format(i, n_train, torch.stack(running_loss).mean(), 100*torch.stack(running_acc).mean(), time.time()-start), end='\r')
            i += 1
            if i > 100:
                break
        print()
        eval(model)
        model.train()

    print('Fine tuning all layers')
    # new dataset batch_size
    train_ds = DataLoader(train, batch_size=batch_size_ft, num_workers=10, shuffle=True)
    n_train = len(train_ds)

    # train all model
    for p in model.parameters():
        p.requires_grad = True

    # new optim and sched
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, nesterov=True, weight_decay=0.0001)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, eta_min=0.00001)

    running_loss = []
    running_acc = []
    start = time.time()
    i = 1
    for imgs, lbls in train_ds:
        imgs = imgs.to(device)
        lbls = lbls.to(device)

        optimizer.zero_grad()

        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        loss2 = criterion2(outputs, lbls)
        loss = loss + loss2

        loss.backward()
        optimizer.step()
        # print statistics
        running_loss.append(loss.detach().cpu())
        running_acc.append(((outputs.argmax(dim=1) == lbls).sum() / lbls.shape[0]).detach().cpu())

        print('{}/{} loss: {:5.02f} acc: {:5.02f} in {:6.01f}'.format(i, n_train, torch.stack(running_loss).mean(),
                                                                      100 * torch.stack(running_acc).mean(),
                                                                      time.time() - start), end='\r')
        i += 1
    print()
    eval(model)

# eval
if args.eval_pretrained:
    eval(model)







