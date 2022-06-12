import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
torch.cuda.set_device(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 48
Epochs = 20

trans_cifar = torchvision.transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
trans_mnist = torchvision.transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
cifar_10 = torchvision.datasets.CIFAR10(root='./data', train=True, transform=trans_cifar, download=False)
mnist = torchvision.datasets.MNIST(root='./data', train=True, transform=trans_mnist, download=False)
data_loader = DataLoader(mnist, batch_size=batch_size, shuffle=True)

def imshow(img):

    #反归一化，将数据重新映射到0-1之间
    img = img / 2 + 0.5
    plt.figure(figsize=(4,3))
    plt.imshow(np.transpose(img.numpy(), (1,2,0)))
    plt.savefig('mnist.png', dpi=600)
    plt.show()


for i, (images, _) in enumerate(data_loader):

    print(i)
    print(images.numpy().shape)
    imshow(torchvision.utils.make_grid(images))
    break
