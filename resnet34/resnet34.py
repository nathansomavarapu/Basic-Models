import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision
import torchvision.transforms as transforms

import os


class resnet34(nn.Module):

    def __init__(self, num_classes):
        super(resnet34, self).__init__()

        self.l1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, ceil_mode=True)
        )

        self.l2_1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.l2_2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.l2_3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.l2_to_3 = nn.Conv2d(64, 128, 1, stride=2)

        self.l3_1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.l3_2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.l3_3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.l3_4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.l3_to_4 = nn.Conv2d(128, 256, 1, stride=2)

        self.l4_1 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.l4_2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.l4_3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.l4_4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True)
            
        )

        self.l4_5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.l4_6 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.l4_to_5 = nn.Conv2d(256, 512, 1, stride=2)

        self.l5_1 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.l5_2 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.l5_3 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.ap = nn.AvgPool2d(7)
        self.cl = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.l1(x)

        x = self.l2_1(x) + x
        x = self.l2_2(x) + x
        x = self.l2_3(x) + x

        x = self.l3_1(x) + self.l2_to_3(x)
        x = self.l3_2(x) + x
        x = self.l3_3(x) + x
        x = self.l3_4(x) + x

        x = self.l4_1(x) + self.l3_to_4(x)
        x = self.l4_2(x) + x
        x = self.l4_3(x) + x
        x = self.l4_4(x) + x
        x = self.l4_5(x) + x
        x = self.l4_6(x) + x

        x = self.l5_1(x) + self.l4_to_5(x)
        x = self.l5_2(x) + x
        x = self.l5_3(x) + x

        x = self.ap(x)

        x = x.view(x.size(0), -1)
        x = self.cl(x)
        
        return x


def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    batch_size = 16
    train_data = torchvision.datasets.CIFAR10(
        root='../data', train=True, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               shuffle=True, num_workers=2)

    test_data = torchvision.datasets.CIFAR10(
        root='../data', train=False, download=True, transform=transform)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                              shuffle=False, num_workers=2)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    viz_ind = 0
    
    model = resnet34(10)
    epochs = 20
    min_loss = float('inf')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)

    if os.path.exists('resnet34.pt'):
        model.load_state_dict(torch.load('resnet34.pt', map_location=lambda storage, loc: storage))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for i in range(epochs):
        # curr_loss = 0
        # total = 0
        # for j, data in enumerate(train_loader):
        #     img, cl = data
        #     img = img.to(device)
        #     cl = cl.to(device)

        #     optimizer.zero_grad()

        #     pred = model(img)
        #     loss = criterion(pred, cl)
        #     loss.backward()
        #     optimizer.step()

        #     curr_loss += loss.item()

        #     if j % 100 == 99:
        #         print('epoch [%d/%d], img [%d,%d], training loss: %f' % (i, epochs, j, len(train_data), curr_loss/200.0))
        #         curr_loss = 0
        
        val_loss = 0
        correct_num = 0.0
        total = 0

        with torch.no_grad():
            for k, data_v in enumerate(test_loader):
                img_v, cl_v = data_v

                img_v = img_v.to(device)
                cl_v = cl_v.to(device)

                pred_v = model(img_v)

                val_loss += criterion(pred_v, cl_v).item()

                total += cl_v.size(0)

                _, pred_cl_v = torch.max(pred_v.data, 1)
                correct_num += (pred_cl_v == cl_v).sum().item()

                if k == viz_ind:
                    torchvision.utils.save_image(img_v, 'test_imgs.png', nrow=4)
                    cl_str = ''
                    for pred_cl_i in pred_cl_v:
                        cl_str += classes[pred_cl_i] + '  '
                    print(cl_str)
        
        if val_loss < min_loss:
            torch.save(model.state_dict(), 'resnet34.pt')
            min_loss = val_loss
        
        print(pred_v)
        
        print('epoch [%d/%d], test loss: %f, percent correct: %f' % (i, epochs, val_loss/len(test_data), correct_num/len(test_data)))

if __name__ == '__main__':
    main()