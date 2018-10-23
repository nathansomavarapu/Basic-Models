import torch
import torch.nn as nn

import torchvision
from torchvision.utils import save_image
from torchvision import datasets

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim

import torchvision.transforms as transforms

import math

class autoenc(nn.Module):
    def __init__(self, num_hidden):
        super(autoenc, self).__init__()


        self.enc = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_hidden),
            nn.ReLU(inplace=True),
        )

        self.dec = nn.Sequential(
            nn.Linear(num_hidden, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 28*28),
            nn.ReLU(inplace=True)
        )

        self._init_weights()
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        encoded = self.enc(x)
        decoded = self.dec(encoded)
        decoded = decoded.view(decoded.size(0), 1, 28, 28)

        return encoded, decoded
    
    def _init_weights(self):
        for layer in self.enc.children():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, 0.01)
                nn.init.constant_(layer.bias, 0)
        
        for layer in self.dec.children():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, 0.01)
                nn.init.constant_(layer.bias, 0)


def main():
    
    trainset = datasets.MNIST('../data/', train=True, download=True, transform=transforms.ToTensor())
    trainloader = DataLoader(trainset, shuffle=True, batch_size=64)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = 20

    enc_dim = 4
    model = autoenc(enc_dim)

    opt = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    img_num = 0
    

    for e in range(epochs):
        for i, data in enumerate(trainloader):
            img, _= data
            img = img.to(device)

            opt.zero_grad()

            encoded, decoded = model(img)
            loss = criterion(decoded, img)

            loss.backward()
            opt.step()

            print('Epoch [%d/%d], Batch [%d/%d], Loss [%f] ' % (e, epochs, i, len(trainloader), loss.item()))

            num = e * len(trainloader) + i
            if num % 1000 == 0:
                encoded = encoded.view(encoded.size(0), 1, math.sqrt(enc_dim), math.sqrt(enc_dim))
                save_image(decoded.data[:25], 'images/%d.png' % img_num, nrow=5)
                save_image(encoded.data[:25], 'images/%d_enc.png' % img_num, nrow=5)
                img_num += 1

if __name__ == '__main__':
    main()
