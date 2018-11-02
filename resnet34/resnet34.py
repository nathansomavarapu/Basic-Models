import torch
import torch.nn as nn
import torchvision.datasets as datasets


class resnet34(nn.Module):

    def __init__(self):
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
        self.cl = nn.Linear(512, 1000)
    
    def forward(self, x):
        x = self.l1(x)

        x = self.l2_1(x) + x
        x = self.l2_2(x) + x
        x = self.l2_3(x) + x

        x = self.l3_1(x)
        x = self.l3_2(x) + x
        x = self.l3_3(x) + x
        x = self.l3_4(x) + x

        x = self.l4_1(x)
        x = self.l4_2(x) + x
        x = self.l4_3(x) + x
        x = self.l4_4(x) + x
        x = self.l4_5(x) + x
        x = self.l4_6(x) + x

        x = self.l5_1(x) 
        x = self.l5_2(x) + x
        x = self.l5_3(x) + x

        x = self.ap(x)

        x = x.view(x.size(0), -1)
        x = self.cl(x)
        
        return x


def main():
    model = resnet34()
    print(model)

    print(model(torch.rand(1, 3, 224, 224)).size())

if __name__ == '__main__':
    main()