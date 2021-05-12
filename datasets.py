import torch
from torch import nn
import torchvision
from torchvision import transforms
import numpy as np

# put all dataset on gpu :)
def build_dataset(device="cuda"):
    _CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD)
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    img = []
    lbl = []
    for i, l in trainset:
        img.append(i.unsqueeze(0))
        lbl.append(l)
    train_imgs = torch.cat(img).half()
    train_lbls = torch.Tensor(lbl).long().to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD)
    ])
    valset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform)

    img = []
    lbl = []
    for i, l in valset:
        img.append(i.unsqueeze(0))
        lbl.append(l)
    val_imgs = torch.cat(img).to(device).half()
    val_lbls = torch.Tensor(lbl).long().to(device)

    return train_imgs, train_lbls, val_imgs, val_lbls

class RandomCrop(nn.Module):
    def __init__(self, size=32, pad=4):
        super(RandomCrop, self).__init__()
        self.size = size
        self.pad = pad
    def forward(self, x):
        i = torch.randint( 2 *self.pad, (2,)).to(x.device).long()
        return x[:, :, i[0]:i[0 ] +self.size, i[1]:i[1 ] +self.size]

class RandomHorizontalFlip(nn.Module):
    def __init__(self):
        super(RandomHorizontalFlip, self).__init__()
    def forward(self, x):
        r = torch.randn((x.shape[0], 1, 1, 1), device=x.device) < 0.
        return r* x + (~r) * x.flip(-1)


class Cutout(nn.Module):
    def __init__(self, height, width):
        super(Cutout, self).__init__()
        self.height = height
        self.width = width

    def __call__(self, image):
        h, w = image.shape[2], image.shape[3]
        mask = np.ones((1, 1, h, w), np.float32)
        y = np.random.choice(range(h))
        x = np.random.choice(range(w))

        y1 = np.clip(y - self.height // 2, 0, h)
        y2 = np.clip(y + self.height // 2, 0, h)
        x1 = np.clip(x - self.width // 2, 0, w)
        x2 = np.clip(x + self.width // 2, 0, w)

        mask[:, :, y1:y2, x1:x2] = 0.
        mask = torch.from_numpy(mask).to(device=image.device, dtype=image.dtype)
        mask = mask.expand_as(image)
        image *= mask
        return image


# run data augmentation as a module on gpu
class Augment(nn.Module):
    def __init__(self):
        super(Augment, self).__init__()
        t = torch.nn.Sequential(
            transforms.RandomCrop(32, (4, 4)),
            transforms.RandomHorizontalFlip(),
            Cutout(8, 8)
        )
        self.transforms = t  # torch.jit.script(t)

    def forward(self, x):
        x = self.transforms(x)
        return x
