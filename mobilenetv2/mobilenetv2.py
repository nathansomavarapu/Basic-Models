import torch
import torch.nn as nn

class mobilenetv2(nn.Module):

    def __init__(self, num_classes, init_weights=True):
        super(mobilenetv2, self).__init__()

        spec = [[1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1]]

        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6()
        )

        self.blocks = self._build_blocks(spec)

        self.conv1 = nn.Sequential(
            nn.Conv2d(320, 1280, 1),
            nn.BatchNorm2d(1280),
            nn.ReLU6()
        ) 
        self.avg = nn.AvgPool2d(7)

        self.linear = nn.Linear(1280, num_classes)

        if init_weights:
            self._init_weights()
    
    def _build_blocks(self, spec, in_channels=32):

        all_blocks = []
        prev_channels = in_channels
        for t, c, n, s in spec:
            
            for j in range(n):
                if j == 1:
                    all_blocks.append(bottleneck(t, prev_channels, c, s))
                else:
                    all_blocks.append(bottleneck(t, prev_channels, c, 1))
                
                prev_channels = c
        
        return nn.Sequential(*all_blocks)

    
    def forward(self, x):
        out = self.conv0(x)

        out = self.blocks(out)
        out = self.conv1(out)
        out = self.avg(out)
        out = out.view(-1, 1280)
        out = self.linear(out)

        return out

    def _init_weights(self):
        

        for layer in self.children():
            if isinstance(layer, nn.Sequential):
                for sublayer in layer:
                    if isinstance(sublayer, bottleneck):
                        sublayer._init_weights()
                    else:
                        shared_weight_init(sublayer)
            else:
                shared_weight_init(layer)

class bottleneck(nn.Module):

    def __init__(self, t, cin, cout, s):

        super(bottleneck, self).__init__()

        expansion = cin * t
        self.stride = s
        self.cin = cin
        self.cout = cout

        self.bottleneck_block = nn.Sequential(
            nn.Conv2d(cin, expansion, 1),
            nn.BatchNorm2d(expansion),
            nn.ReLU6(),
            nn.Conv2d(expansion, expansion, 3, stride=s, groups=expansion, padding=1),
            nn.BatchNorm2d(expansion),
            nn.ReLU6(),
            nn.Conv2d(expansion, cout, 1)
        )

        self.skip = nn.Sequential(
            nn.Conv2d(cin, cout, 1),
            nn.BatchNorm2d(cout)
        )
    
    def forward(self, x):

        if self.stride == 1:
            if self.cin != self.cout:
                x_n = self.skip(x)
            else:
                x_n = x
            return self.bottleneck_block(x) + x_n
        else:
            return self.bottleneck_block(x)

    def _init_weights(self):
        for layer in self.children():
            shared_weight_init(layer)

def shared_weight_init(layer):
    if isinstance(layer, nn.Conv2d):
        nn.init.normal_(layer.weight)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)
    if isinstance(layer, nn.BatchNorm2d):
        nn.init.constant_(layer.weight, 1)
        nn.init.constant_(layer.bias, 0)
    if isinstance(layer, nn.Linear):
        nn.init.normal_(layer.weight, std=1e-3)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)
            