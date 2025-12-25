import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
from torch.nn import functional as F
from .CBAM import CBAM


__all__ = ['BL', 'FD_BL']
model_urls = {
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}

class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = F.upsample_bilinear(x, scale_factor=2)
        x = self.reg_layer(x)
        return torch.abs(x)


class VGG_FD(nn.Module):
    def __init__(self, features):
        super(VGG_FD, self).__init__()
        self.decoded_component = 'common'
        
        self.features = features
        self.decomposer_common = CBAM(512)
        self.decomposer_unique = CBAM(512)

        if self.decoded_component == 'concat':
            self.linear = nn.Conv2d(1024, 512, 1)
        
        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )

    def forward(self, x_high, x_low=None):
        feat_high = self.features(x_high)   # vgg network
        
        feat_high_common = self.decomposer_common(feat_high)
        feat_high_unique = self.decomposer_unique(feat_high)
        
        if self.decoded_component == 'common':
            x_high = feat_high_common
        elif self.decoded_component == 'unique':
            x_high = feat_high_unique
        elif self.decoded_component == 'concat':
            x_high = self.linear(torch.cat([feat_high_common, feat_high_unique], 1))
        
        x_high = F.upsample_bilinear(x_high, scale_factor=2)
        x_high = self.reg_layer(x_high)
        
        if x_low is not None:
            feat_low = self.features(x_low)   # vgg network
        
            feat_low_common = self.decomposer_common(feat_low)
            feat_low_unique = self.decomposer_unique(feat_low)
            
            if self.decoded_component == 'common':
                x_low = feat_low_common
            elif self.decoded_component == 'unique':
                x_low = feat_low_unique
            elif self.decoded_component == 'concat':
                x_low = self.linear(torch.cat([feat_low_common, feat_low_unique], 1))

            x_low = F.upsample_bilinear(x_low, scale_factor=2)
            x_low = self.reg_layer(x_low)
            return feat_high, feat_high_common, feat_high_unique, torch.abs(x_high), \
                    feat_low, feat_low_common, feat_low_unique, torch.abs(x_low)
                    
        return torch.abs(x_high)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
}

def BL():
    """VGG 19-layer model (configuration "E")
        model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E']))
    model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
    return model
    
def FD_BL():
    """VGG 19-layer model (configuration "E")
        model pre-trained on ImageNet
    """
    model = VGG_FD(make_layers(cfg['E']))
    model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
    return model