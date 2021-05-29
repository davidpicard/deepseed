import json

import numpy as np
from torch import optim
import time
import argparse
import torchsummaryX
from collections import OrderedDict
from torch.utils.data import DataLoader

from datasets import *
from loss import *

batch_size = 128
batch_size_ft = 128
v_batch_size = 50
epoch = 1
ft_epoch = 50
max_train = 4000
max_train_ft = 24000

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
    return torch.stack(val_loss).mean().item(), 100. * torch.stack(val_acc).mean().item()


#############


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", help="path to imagenet")
parser.add_argument("--eval_pretrained", type=bool, default=False)
parser.add_argument("--output", default="results_imagenet.json")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--nb_seeds", type=int, default=10)
args = parser.parse_args()

data = []

train, val = build_imagenet(args.data_dir)
val_ds = DataLoader(val, batch_size=v_batch_size, num_workers=10, pin_memory=True)
n_val = len(val_ds)

for s in range(args.seed, args.seed + args.nb_seeds):
    print('doing seed {}'.format(s))
    torch.manual_seed(s)
    np.random.seed(s)


    tr_loss = []
    tr_acc = []
    ft_loss = []
    ft_acc = []

    # build model

    model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
    # model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet50_ssl')
    model.fc = nn.Linear(2048, 1000)
    if not args.eval_pretrained:
        torch.nn.init.kaiming_uniform_(model.fc.weight)
        torch.nn.init.normal_(model.fc.bias)
    # torchsummaryX.summary(model, torch.zeros((1, 3, 224, 224)))

    for p in model.parameters():
        p.requires_grad = False
    if not args.eval_pretrained:
        model.fc.weight.requires_grad = True
        model.fc.bias.requires_grad = True

    model.to(device)

    train_ds = DataLoader(train, batch_size=batch_size, num_workers=12, shuffle=True, pin_memory=True)
    n_train = len(train_ds)

    # optimization hparams
    criterion = nn.CrossEntropyLoss()

    if not args.eval_pretrained:

        criterion2 = CrossEntropyLabelSmooth(num_classes=1000, epsilon=0.1)
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=0.0001)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_train, eta_min=0.001)

        # training loop
        print('Training last layer')
        t_start = time.time()
        i = 1
        for e in range(epoch):  # loop over the dataset multiple times
            running_loss = []
            running_acc = []
            start = time.time()
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

                if i%1000 == 0:
                    print()
                    l, a = eval(model)
                    tr_loss.append(l)
                    tr_acc.append(a)
                    model.train()
                sched.step()

                if i >= max_train:
                    break

                i += 1

        print('Fine tuning all layers')
        # new dataset batch_size
        train_ds = DataLoader(train, batch_size=batch_size_ft, num_workers=10, shuffle=True, pin_memory=True)
        n_train = len(train_ds)

        # train all model
        for p in model.parameters():
            p.requires_grad = True

        # new optim and sched
        optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.9, nesterov=True, weight_decay=0.0001)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_train_ft, eta_min=0.0001)

        i = 1
        for e in range(ft_epoch):
            running_loss = []
            running_acc = []
            for imgs, lbls in train_ds:
                imgs = imgs.to(device)
                lbls = lbls.to(device)

                optimizer.zero_grad()

                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                loss = loss

                loss.backward()
                optimizer.step()
                # print statistics
                running_loss.append(loss.detach().cpu())
                running_acc.append(((outputs.argmax(dim=1) == lbls).sum() / lbls.shape[0]).detach().cpu())

                print('{}/{} loss: {:5.02f} acc: {:5.02f} in {:6.01f}'.format(i, n_train, torch.stack(running_loss).mean(),
                                                                              100 * torch.stack(running_acc).mean(),
                                                                              time.time() - start), end='\r')

                if i % 2000 == 0:
                    print()
                    l, a = eval(model)
                    ft_loss.append(l)
                    ft_acc.append(a)
                    model.train()
                sched.step()

                if i >= max_train_ft:
                    break
                i += 1

    # eval
    if args.eval_pretrained:
        eval(model)
    d = { 'seed': s,
        'tr_loss': tr_loss,
        'tr_acc': tr_acc,
        'ft_loss': ft_loss,
        'ft_acc': ft_acc,
        'time': time.time() - start
        }

    data.append(d)
    print(d)
    with open(args.output, 'w') as fp:
        json.dump(data, fp)






