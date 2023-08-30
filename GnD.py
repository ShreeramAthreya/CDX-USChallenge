import torch
from torch import nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, num_features):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(num_features))

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        return out

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.conv1 = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
        nn.InstanceNorm2d(64),
        nn.ReLU(inplace=True))
        
        self.conv2 = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, padding_mode='reflect'),
        nn.InstanceNorm2d(128),
        nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(
        nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, padding_mode='reflect'),
        nn.InstanceNorm2d(256),
        nn.ReLU(inplace=True))
        
        self.res_blocks = nn.ModuleList([ResBlock(256) for _ in range(15)])

        self.conv4 = nn.Sequential(
        nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
        nn.InstanceNorm2d(128),
        nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
        nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
        nn.InstanceNorm2d(64),
        nn.ReLU(inplace=True))

        self.conv6 = nn.Conv2d(128, 1, kernel_size=7, stride=1, padding=3, padding_mode='reflect')

    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x = self.conv3(x2)    

        # Residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)

        # Decoder
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv4(x)
        x = torch.cat([x, x2], dim=1)

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv5(x)
        x = torch.cat([x, x1], dim=1)
        x = torch.tanh(self.conv6(x))
        
        return x


# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, padding_mode='reflect')),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.norm1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, padding_mode='reflect')),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.norm2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, padding_mode='reflect')),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.norm3 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, padding_mode='reflect')),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, padding_mode='reflect')

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.norm2(x)
        x = self.norm3(x)
        x = self.conv2(x)

        return x
