import torch
import torch.nn as nn
import torchvision

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim

import numpy as np


class Generator(nn.Module):

    def __init__(self, latent_dim=100):

        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.model(z)
        x = x.view(x.size(0), 1, 28, 28)
        return x


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), 784)
        x = self.model(x)
        return x


def main():

    dataset = datasets.MNIST('../data/', train=True, download=True, transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
    trainloader = DataLoader(dataset, shuffle=True, batch_size=64)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = 15

    D = Discriminator()
    G = Generator()

    D = D.to(device)
    G = G.to(device)

    # print(G, D)

    optim_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optim_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))

    criterion = nn.BCELoss()

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    img_num = 0
    for e in range(epochs):
        for i, data in enumerate(trainloader):
            img, _ = data
            img = img.to(device)

            z = Variable(Tensor(np.random.normal(0, 1, (img.size(0), 100))))

            real = Tensor(np.ones((img.size(0), 1)))
            fake = Tensor(np.zeros((img.size(0), 1)))

            generated_imgs = G(z)

            optim_G.zero_grad()

            generator_loss = criterion(D(generated_imgs), real)
            generator_loss.backward()
            optim_G.step()

            optim_D.zero_grad()

            discriminator_loss1 = criterion(D(generated_imgs.detach()), fake)
            discriminator_loss2 = criterion(D(img), real)


            d_loss_total = (discriminator_loss1 + discriminator_loss2)/2
            d_loss_total.backward()
            optim_D.step()
            
            print('Epoch [%d/%d], Batch [%d/%d], Disc Loss [%f], Gen Loss [%f]' % (e, epochs, i, len(trainloader), d_loss_total.item(), generator_loss.item()))

            num = e * len(trainloader) + i
            if num % 400 == 0:
                save_image(generated_imgs.data[:25], 'images/%d.png' % img_num, nrow=5, normalize=True)
                img_num += 1


if __name__ == '__main__':
    main()
