import torch
from torch.utils.data import DataLoader
import torch.nn as nn

import torchvision
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Resize, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, ToTensor, Normalize

def train_model(model, epochs, trainloader, crit, opt, device, testloader=None, verbose=False, sched=None):
    """Generic model training interface.
    
    Arguments:
        model {Torch model} -- Classification model, adjust the model to fit the input from the datalaoder
        trainloader {Torch Dataloader} --  Dataloader for training data
        crit {Torch Loss} -- Loss Criterion
        opt {Torch Optimizer} -- Optimizer
        device {Torch Device} -- Location to send tensors
    
    Keyword Arguments:
        testloader {Torch Dataloader} -- Dataloader for testing data (default: {None})
    """
    assert epochs != 0

    min_test_loss = float('inf')
    max_top1 = 0.0
    
    for e in range(epochs):
        if verbose:
            print('Epoch [%d/%d]' % (e, epochs))
            print('----------------------------------')

        model.train()
        run_inference(model, trainloader, crit, opt, device, verbose=verbose, sched=sched)

        if testloader is not None:
            if verbose: 
                print('\n')
                print('Testing')
            
            model.eval()
            with torch.no_grad():
                test_loss, top1 = run_inference(model, testloader, crit, opt, device, verbose=verbose, train=False)
                
                min_test_loss = min(min_test_loss, test_loss)
                max_top1 = max(max_top1, top1)

                if max_top1 == top1:
                    torch.save(model.state_dict, 'model.pt')
            
            if verbose:
                print('Test Loss: %f, Test Top1 %f' % (test_loss, top1))
        
    print('Min Test Loss: %f, Max Test Top1 %f' % (min_test_loss, max_top1))


def run_inference(model, loader, crit, opt, device, verbose=False, train=True, sched=None):

    total_loss = 0.0
    preds = []
    targets = []

    for i, data in enumerate(loader):
        imgs, labels = data
        imgs = imgs.to(device)
        labels = labels.to(device)

        out = model(imgs)
        loss = crit(out, labels)

        preds.append(torch.max(out, dim=1)[1])
        targets.append(labels)

        if train:

            opt.zero_grad()
            loss.backward()
            opt.step()

        total_loss += loss.item()

        if verbose:
            print('Iteration [%d/%d], Loss: %f' % (i, len(loader), loss.item()))
    
    if train and sched is not None:
        sched.step()
    
    preds = torch.cat(preds)
    targets = torch.cat(targets)

    n = preds.size()[0]

    total_loss = total_loss / float(n)
    top1 = torch.sum((preds == targets)).item()/ float(n)
        
    return total_loss, top1

def train_cifar(model, epochs, opt, verbose=False, sched=None):

    transforms_train = Compose([
        Resize(224),
        RandomCrop(224),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ])

    transforms_test = Compose([
        Resize(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    trainset = CIFAR10('../data/', train=True, transform=transforms_train, download=True)
    trainloader = DataLoader(trainset, batch_size=96, shuffle=True)

    testset = CIFAR10('../data/', train=False, transform=transforms_test, download=True)
    testloader = DataLoader(testset, batch_size=100, shuffle=True)

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    crit = nn.CrossEntropyLoss()

    model = model.to(device)

    train_model(model, epochs, trainloader, crit, opt, device, testloader=testloader, verbose=verbose, sched=sched)
