import torch
from torch import optim
import time

from models import *
from datasets import *
from loss import *

batch_size = 500
v_batch_size = 100
epoch = 22

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

train_imgs, train_lbls, val_imgs, val_lbls = build_dataset(device=device)
n_train = len(train_lbls)
n_val = len(val_lbls)


data = []
for s in range(5):
  torch.manual_seed(s)
  np.random.seed(s)

  net = build_network()
  net.to(device).half()
  for layer in net.modules():
    if isinstance(layer, nn.BatchNorm2d):
      layer.float()
      if hasattr(layer, 'weight') and layer.weight is not None:
        layer.weight.data.fill_(1.0)
      layer.eps = 0.00001
      layer.momentum = 0.1

  criterion = nn.CrossEntropyLoss()
  criterion2 = CrossEntropyLabelSmooth(num_classes=10, epsilon=0.2)
  optimizer = optim.SGD(net.parameters(), lr=0.2, momentum=0.9, nesterov=True, weight_decay=0.001)

  def lr(e):
    if e < 4:
      return 0.5*e/3. + 0.01
    return 0.5*(22-e)/19. + 0.01
  sched = optim.lr_scheduler.LambdaLR(optimizer, lr)

  augment = Augment()
  augment.to(device).half()

  tl = []
  vl = []
  va = []
  tt = []
  t_start = time.time()
  for e in range(epoch):  # loop over the dataset multiple times
    start = time.time()

    # process training set
    a_train = []
    for i in range(n_train//batch_size):
      # get the inputs; data is a list of [inputs, labels]
      inputs = train_imgs[i*batch_size:(i+1)*batch_size, ...]
      a_train.append(augment(inputs.to(device).half()))
    a_train_imgs = torch.cat(a_train)
    perm = torch.randperm(n_train)
    a_train_imgs = a_train_imgs[perm, ...].contiguous()
    a_train_lbls = train_lbls[perm].contiguous()

    # a_stop = time.time()

    net.train()
    running_loss = []
    perm = torch.randperm(n_train)
    # t1 = 0
    # t2 = 0
    # t3 = 0
    for i in range(n_train//batch_size):
      # s = time.time()
      # get the inputs; data is a list of [inputs, labels]
      inputs = a_train_imgs[i*batch_size: (i+1)*batch_size, ...]
      labels = a_train_lbls[i*batch_size: (i+1)*batch_size]

      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      outputs = net(inputs)
      loss = criterion(outputs, labels)
      loss2 = criterion2(outputs, labels)
      loss = loss + 2*loss2
      # torch.cuda.synchronize()
      # t1 += time.time() - s
      loss.backward()
      # torch.cuda.synchronize()
      # t2 += time.time() - s
      optimizer.step()
      # torch.cuda.synchronize()
      # t3 += time.time() - s

      # print statistics
      running_loss.append(loss)
    running_loss = torch.stack(running_loss).mean().item()
    # t_stop = time.time()
    # t1 /= n_train//batch_size
    # t2 /= n_train//batch_size
    # t3 /= n_train//batch_size

    if e == 0 or e%5 == 1:
      net.eval()
      val_loss = []
      val_acc = []
      for i in range(n_val//v_batch_size):
        # get the inputs; data is a list of [inputs, labels]
        inputs = val_imgs[i*v_batch_size: (i+1)*v_batch_size, ...]
        labels = val_lbls[i*v_batch_size: (i+1)*v_batch_size]
        outputs = net(inputs)
        val_loss.append(criterion(outputs, labels))
        val_acc.append((outputs.argmax(dim=1) == labels).sum()/labels.shape[0])

      v_stop = time.time()
      # print('{} train loss {:5.02f} val loss {:5.02f} val acc {:5.02f} time a:{:5.03f} t:{:5.03f}, v:{:5.03f}, t1:{:5.03f}, t2:{:5.03f}, t3:{:5.03f} '.format(
        # e, running_loss, torch.stack(val_loss).mean(), 100.*torch.stack(val_acc).mean(), (a_stop-start), (t_stop-start), (v_stop - start), t1, t2, t3))
      # print('{} train loss {:5.02f} val loss {:5.02f} val acc {:5.02f} time v:{:5.03f}'.format(
      #   e, running_loss, torch.stack(val_loss).mean(), 100.*torch.stack(val_acc).mean(), (v_stop - start)))
      tl.append(running_loss)
      vl.append(torch.stack(val_loss).mean().item())
      va.append(100.*torch.stack(val_acc).mean().item())
      tt.append(v_stop - start)
    sched.step()
  d = { 'train_loss': tl,
        'val_loss': vl,
        'val_acc': va,
        'time': tt}
  print('Finished Training in {:5.03f}  train loss {:5.02f} val loss {:5.02f} val acc {:5.02f}'.format(time.time()-t_start, tl[-1], vl[-1], va[-1]))
  data.append(d)

print(data)


