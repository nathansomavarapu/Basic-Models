import sys
sys.path.append('../runner/')

from runner import train_model, train_cifar

from mobilenetv2 import mobilenetv2
from torchvision.models import resnet50

import torch
import torch.nn as nn
import torch.optim as optim

model = mobilenetv2(10, init_weights=False)

epochs = 350
opt = optim.SGD(model.parameters(), lr=0.1)
sched = optim.lr_scheduler.StepLR(opt, step_size=150, gamma=0.1)
# opt = optim.RMSprop(model.parameters(), lr=0.045)
# opt = optim.Adam(model.parameters())

train_cifar(model, epochs, opt, verbose=True, sched=sched)




