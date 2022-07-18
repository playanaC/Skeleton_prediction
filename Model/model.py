""" Full assembly of the parts to form the complete network """
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
        

# *******************************************************************************************************
# *******************************************************************************************************
class Encoder_skl(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(Encoder_skl, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

    def forward(self, y):
        y = self.inc(y)
        y = self.down1(y)
        y = self.down2(y)
        y = self.down3(y)
        y = self.down4(y)
        return y


class Model_AL(nn.Module):
    def __init__(self, n_seq, in_size, hidden_size, batch_size, n_channels, n_classes, bilinear=True):
        super(Model_AL, self).__init__()
        self.in_size = in_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.in_size = in_size
        self.n_seq = n_seq

        self.encoder_skl = Encoder_skl(1, bilinear=True)
        input_LSTM = pow(int(in_size / 16), 2) * 512
        self.ULSTM = nn.LSTM(input_size=input_LSTM, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size * n_seq, input_LSTM)

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024*2, 512, bilinear)
        self.up2 = Up(512*2, 256, bilinear)
        self.up3 = Up(256*2, 128, bilinear)
        self.up4 = Up(128*2, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x, y):
        x1 = self.inc(x[:, 1, :, :].unsqueeze(0).permute(1, 0, 2, 3))
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        z1 = self.inc(x[:, 0, :, :].unsqueeze(0).permute(1, 0, 2, 3))
        z2 = self.down1(z1)
        z3 = self.down2(z2)
        z4 = self.down3(z3)
        z5 = self.down4(z4)

        z5 = torch.cat([x5, z5], dim=1)
        z5 = z5.view(self.batch_size, self.n_seq, -1)
        z5 = self.ULSTM(z5)[0]
        z5 = z5.reshape(self.batch_size, self.n_seq * self.hidden_size)
        z5 = self.fc(z5)
        z5 = z5.reshape(x5.size())

        y = self.encoder_skl(y)
        y = torch.cat([z5, y], dim=1)

        x4 = torch.cat([x4, z4], dim=1)
        x = self.up1(y, x4)

        x3 = torch.cat([x3, z3], dim=1)
        x = self.up2(x, x3)

        x2 = torch.cat([x2, z2], dim=1)
        x = self.up3(x, x2)

        x1 = torch.cat([x1, z1], dim=1)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x
