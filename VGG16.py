import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class VGG16(nn.Module):

    def __init__(self, num_classes):
        super(VGG16, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.mp = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 128, 3)
        # self.mp

        self.conv5 = nn.Conv2d(128, 256, 3)
        self.conv6 = nn.Conv2d(256, 256, 3)
        self.conv7 = nn.Conv2d(256, 256, 3)
        # self.mp

        self.conv8 = nn.Conv2d(256, 512, 3)
        self.conv9 = nn.Conv2d(512, 512, 3)
        self.conv10 = nn.Conv2d(512, 512, 3)
        # self.mp

        self.conv11 = nn.Conv2d(512, 512, 3)
        self.conv12 = nn.Conv2d(512, 512, 3)
        self.conv13 = nn.Conv2d(512, 512, 3)
        # self.mp

        self.linear1 = nn.Linear(512, 4096)
        self.linear2 = nn.Linear(4096, 4096)
        self.linear3 = nn.Linear(4096, num_classes)

        for l in self.children():
            if isinstance(l, nn.Conv2d):
                nn.init.kaiming_normal_(l.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(l, nn.Linear):
                nn.init.normal_(l.weight, 0, 0.01)
                nn.init.constant_(l.bias, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.mp(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.mp(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = self.mp(x)

        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = self.mp(x)

        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x = self.mp(x)

        x = x.view(-1, 512)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))

        return x


def main():

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_data = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=4,
                                               shuffle=False, num_workers=2)

    test_data = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=4,
                                              shuffle=False, num_workers=2)

    model = VGG16(10)
    epochs = 20
    min_loss = float('inf')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for i in range(epochs):
        
        curr_loss = 0
        for j, data in enumerate(train_loader):
            img, cl = data
            img = img.to(device)
            cl = cl.to(device)

            optimizer.zero_grad()

            pred = model(img)
            loss = criterion(pred, cl)
            loss.backward()
            optimizer.step()

            curr_loss += loss.item()

            if j % 2000 == 1999:
                print('epoch [%d/%d], img [%d,%d], training loss: %f' % format(i, epochs, j, len(train_data), curr_loss/2000.0))
        
        val_loss = 0
        with torch.set_grad_enabled(False):
            for data_v in test_loader:
                img_v, cl_v = data_v

                img_v = img_v.to(device)
                cl_v = cl_v.to(device)

                pred_v = model(img)

                val_loss += criterion(pred_v, cl_v)
        
        if val_loss < min_loss:
            model.save_state_dict('VGG16.pt')
        
        print('epoch [%d/%d] test loss: %f' % format(i, len(epochs), val_loss/len(test_data)))


if __name__ == '__main__':
    main()
