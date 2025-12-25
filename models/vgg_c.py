import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
from torch.nn import functional as F
from .transformer_cosine import TransformerEncoder, TransformerEncoderLayer
from .CBAM import CBAM


__all__ = ['MAN', 'FD_MAN']
model_urls = {'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'}

class VGG_Trans(nn.Module):
    def __init__(self, features):
        super(VGG_Trans, self).__init__()
        self.features = features

        d_model = 512
        nhead = 2
        num_layers = 4
        dim_feedforward = 2048
        dropout = 0.1
        activation = "relu"
        normalize_before = False
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        if_norm = nn.LayerNorm(d_model) if normalize_before else None

        self.encoder = TransformerEncoder(encoder_layer, num_layers, if_norm)
        self.reg_layer_0 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        rh = int(h) // 16
        rw = int(w) // 16
        x = self.features(x)   # vgg network

        bs, c, h, w = x.shape
        x = x.flatten(2).permute(2, 0, 1)
        x, features = self.encoder(x, (h,w))   # transformer
        x = x.permute(1, 2, 0).view(bs, c, h, w)
        
        x = F.upsample_bilinear(x, size=(rh, rw))
        x = self.reg_layer_0(x)   # regression head
        return torch.relu(x), features


class VGG_Trans_FD(nn.Module):
    def __init__(self, features):
        super(VGG_Trans_FD, self).__init__()
        self.decoded_component = 'common'
        
        self.features = features
        self.decomposer_common = CBAM(512)
        self.decomposer_unique = CBAM(512)
        
        if self.decoded_component == 'concat':
            self.linear = nn.Conv2d(1024, 512, 1)
            
        d_model = 512
        nhead = 2
        num_layers = 4
        dim_feedforward = 2048
        dropout = 0.1
        activation = "relu"
        normalize_before = False
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        if_norm = nn.LayerNorm(d_model) if normalize_before else None

        self.encoder = TransformerEncoder(encoder_layer, num_layers, if_norm)
        self.reg_layer_0 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )
        
    def forward(self, x_high, x_low=None):
        b, c, h, w = x_high.shape
        rh = int(h) // 16
        rw = int(w) // 16
        feat_high = self.features(x_high)   # vgg network
        
        feat_high_common = self.decomposer_common(feat_high)
        feat_high_unique = self.decomposer_unique(feat_high)
        
        if self.decoded_component == 'common':
            x_high = feat_high_common
        elif self.decoded_component == 'unique':
            x_high = feat_high_unique
        elif self.decoded_component == 'concat':
            x_high = self.linear(torch.cat([feat_high_common, feat_high_unique], 1))

        bs, c, h, w = x_high.shape
        x_high = x_high.flatten(2).permute(2, 0, 1)
        x_high, features_high = self.encoder(x_high, (h,w))   # transformer
        x_high = x_high.permute(1, 2, 0).view(bs, c, h, w)
        
        x_high = F.upsample_bilinear(x_high, size=(rh, rw))
        x_high = self.reg_layer_0(x_high)   # regression head
        
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

            x_low = x_low.flatten(2).permute(2, 0, 1)
            x_low, features_low = self.encoder(x_low, (h,w))   # transformer
            x_low = x_low.permute(1, 2, 0).view(bs, c, h, w)
            
            x_low = F.upsample_bilinear(x_low, size=(rh, rw))
            x_low = self.reg_layer_0(x_low)   # regression head
            return feat_high, feat_high_common, feat_high_unique, torch.relu(x_high), features_high, \
                    feat_low, feat_low_common, feat_low_unique, torch.relu(x_low), features_low
        
        return torch.relu(x_high), features_high


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
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

def MAN():
    """VGG 19-layer model (configuration "E")
        model pre-trained on ImageNet
    """
    model = VGG_Trans(make_layers(cfg['E']))
    model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
    return model
    
def FD_MAN():
    """VGG 19-layer model (configuration "E")
        model pre-trained on ImageNet
    """
    model = VGG_Trans_FD(make_layers(cfg['E']))
    model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
    return model