import torch
import torch.nn as nn

class mobilenetv2(nn.Module):

    def __init__(self, num_classes):
        super(mobilenetv2, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU6()
        )

        self.b0 = bottleneck(1, 32, 16, 1, 1)

        self.b0_b1 = nn.Conv2d(32, 6*16, 1, stride=2)
        self.b1 = bottleneck(6, 16, 24, 2, 2)

        self.b1_b2 = nn.Conv2d(6*16, 6*24, 1, stride=2)
        self.b2 = bottleneck(6, 24, 32, 3, 2)

        self.b2_b3 = nn.Conv2d(6*24, 6*32, 1, stride=2)
        self.b3 = bottleneck(6, 32, 64, 4, 2)

        self.b3_b4 = nn.Conv2d(6*32, 6*64, 1)
        self.b4 = bottleneck(6, 64, 96, 3, 1)

        self.b4_b5 = nn.Conv2d(6*64, 6*96, 1, stride=2)
        self.b5 = bottleneck(6, 96, 160, 3, 2)

        self.b5_b6 = nn.Conv2d(6*96, 6*160, 1)
        self.b6 = bottleneck(6, 160, 320, 1, 1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(320, 1280, 1),
            nn.ReLU6()
        ) 
        self.avg = nn.AvgPool2d(7)

        self.linear = nn.Linear(1280, num_classes)
    
    def forward(self, x):

        x = self.conv0(x)
        
        out, x = self.b0(x)

        x = self.b0_b1(x)
        out, x = self.b1(out, input_residual=x)

        x = self.b1_b2(x)
        out, x = self.b2(out, input_residual=x)

        x = self.b2_b3(x)
        out, x = self.b3(out, input_residual=x)

        x = self.b3_b4(x)
        out, x = self.b4(out, input_residual=x)

        x = self.b4_b5(x)
        out, x = self.b5(out, input_residual=x)

        x = self.b5_b6(x)
        out, x = self.b6(out, input_residual=x)

        out = self.conv1(out)
        out = self.avg(out)
        out = out.view(-1, 1280)
        out = self.linear(out)

        return out

class bottleneck(nn.Module):

    def __init__(self, t, cin, cout, n, s):

        super(bottleneck, self).__init__()

        expansion = cin * t
        self.n = n

        self.bottleneck_block_s = nn.Sequential(
            nn.Conv2d(cin, expansion, 1),
            nn.ReLU6(),
            nn.Conv2d(expansion, expansion, 3, stride=s, padding=1),
            nn.ReLU6()
        )

        self.depth_expansion = nn.Sequential(
            nn.Conv2d(expansion, cout, 1)
        )

        self.bottleneck_block = nn.Sequential(
            nn.Conv2d(cout, expansion, 1),
            nn.ReLU6(),
            nn.Conv2d(expansion, expansion, 3, 1, padding=1),
            nn.ReLU6()
        )
    
    def forward(self, x, input_residual=None):


        if input_residual is None:
            x = self.bottleneck_block_s(x)
            out = self.depth_expansion(x)
        else:
            x = self.bottleneck_block_s(x) + input_residual
            out = self.depth_expansion(x)

        for _ in range(self.n-1):
            x = self.bottleneck_block(out) + x
            out = self.depth_expansion(x)

        return out, x