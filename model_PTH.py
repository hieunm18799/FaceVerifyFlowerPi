import torch
import torchvision.models as models
import torch.nn as nn
# from torchsummary import summary

INPUT_SHAPE = (3, 112, 96)

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_ratio):
        super(Bottleneck, self).__init__()
        self.use_shortcut = (stride == 1 and in_channels == out_channels)
        hidden_dim = int(round(in_channels * expansion_ratio))

        #
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.PReLU(hidden_dim),

            # dw
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.PReLU(hidden_dim),

            # pw-linear
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileFaceNet(nn.Module):
    def __init__(self, in_channels=3):
        super(MobileFaceNet, self).__init__()
        block = Bottleneck
        bottleneck_settings = [
            # t, c , n ,s
            [2, 64, 5, 2],
            [4, 128, 1, 2],
            [2, 128, 6, 1],
            [4, 128, 1, 2],
            [2, 128, 2, 1]
        ]

        self.layers = []
        # conv1
        self.layers.append(nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64)
        ))
        # conv2
        self.layers.append(nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, groups=64, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64)
        ))

        # bottleneck layers
        self.in_channels = 64
        self.layers.extend(self._make_layer(block, bottleneck_settings))

        # conv3
        self.layers.append(nn.Sequential(
            nn.Conv2d(self.in_channels, 512, 1, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.PReLU(512)
        ))

        # GDConv
        self.layers.append(nn.Sequential(
            nn.Conv2d(512, 512, (7, 6), 1, 0, groups=512, bias=False),
            nn.BatchNorm2d(512)
        ))

        # linearConv
        self.layers.append(nn.Sequential(
            nn.Conv2d(512, 128, 1, 1, 0, bias=False),
            nn.BatchNorm2d(128)
        ))

        self.layers = nn.Sequential(*self.layers)

    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(block(self.in_channels, c, stride, t))
                self.in_channels = c
        return layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            # print(x.shape)
        x = x.view(x.size(0), -1)
        return x

# Check if GPU is available and use it for model loading
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = MobileFaceNet().to(device)

# model_path = '/content/gdrive/MyDrive/saved_models/best_model.pth'
# model.load_state_dict(torch.load(model_path, map_location=device))

# summary(model, (3, 112, 96))