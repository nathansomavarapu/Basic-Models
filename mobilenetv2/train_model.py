import sys
sys.path.append('../runner/')

from runner import train_model, train_cifar

from mobilenetv2 import mobilenetv2

import torch
import torch.optim as optim

model = mobilenetv2(10)
epochs = 20
opt = optim.RMSprop(model.parameters())

train_cifar(model, epochs, opt, verbose=True)



