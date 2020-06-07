import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
class basic_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, upsample=1):
        super(basic_block, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.bn = norm_layer(in_channels, eps=1e-3, momentum=0.001)
        self.relu = nn.ReLU(inplace=True)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels,
                                 kernel_size=1, stride=stride)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1)
        self.conv1x3 = nn.Conv2d(out_channels, out_channels,
                                 kernel_size=[1, 3], stride=1, padding=[0, 1])
        self.conv3x1 = nn.Conv2d(out_channels, out_channels,
                                 kernel_size=[3, 1], stride=1, padding=[1, 0])
        self.upsample = None
        if upsample > 1:
            self.upsample = nn.Upsample(scale_factor=upsample,
                                        mode='bilinear',
                                        align_corners=True)

    def forward(self, x):
        preact = self.relu(self.bn(x))
        shortcut = self.conv1x1(preact)
        residual = self.conv3x3(preact)
        residual = self.conv1x3(residual)
        residual = self.conv3x1(residual)
        out = shortcut + residual
        if self.upsample:
            out = self.upsample(out)
        return out

def _upsample_like(src,tar):

    src = F.upsample(src,size=tar.shape[2:],mode='bilinear')

    return src


class JITNET(nn.Module):
    def __init__(self,in_ch=3,out_ch=1,
                 encoder_channels=[8, 32, 64, 64, 128],
                 encoder_strides=[2, 2, 2, 2, 2],
                 decoder_channels=[128, 64, 32, 32, 32],
                 decoder_strides=[1, 1, 1, 1, 1],
                 decoder_upsamples=[2, 2, 4, 1, 2],
                 bn_momentum=0.001,
                 **_):
        super(JITNET, self).__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(3, encoder_channels[0], 3, 2, 1),
            nn.BatchNorm2d(encoder_channels[0], momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(encoder_channels[0], encoder_channels[1], 3, 2, 1),
            nn.BatchNorm2d(encoder_channels[1], momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )

        self.enc_blocks = []
        for i in range(2, len(encoder_channels)):
            self.enc_blocks.append(basic_block(encoder_channels[i - 1],
                                               encoder_channels[i],
                                               encoder_strides[i]))
        self.enc_blocks = nn.ModuleList(self.enc_blocks)
        self.dec_blocks = []
        prev_c = encoder_channels[-1]
        for i in range(0, len(decoder_channels) - 2):
            self.dec_blocks.append(basic_block(prev_c,
                                               decoder_channels[i],
                                               decoder_strides[i],
                                               decoder_upsamples[i]))
            prev_c = decoder_channels[i] + encoder_channels[-i - 2]
        self.dec_blocks = nn.ModuleList(self.dec_blocks)

        self.dec1 = nn.Sequential(
            nn.Conv2d(decoder_channels[-3], decoder_channels[-2], 3, 1, 1),
            nn.BatchNorm2d(decoder_channels[-2], momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(decoder_channels[-2], decoder_channels[-1], 3, 1, 1),
            nn.BatchNorm2d(decoder_channels[-1], momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )

        self.dec_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final = nn.Conv2d(decoder_channels[-1], 1, 1)

        self._initialize_weights()

    def forward(self, x):
        _x = x
        x = self.enc1(x)
        x = self.enc2(x)
        down_x = []
        for b in self.enc_blocks:
            x = b(x)
            down_x.append(x)
        for i, b in enumerate(self.dec_blocks):
            x = b(x)
            if i < len(self.dec_blocks) - 1:
                dx = down_x[-i - 2]
                x = torch.cat([dx, x[:, :, :dx.shape[2], :dx.shape[3]]], dim=1)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec_upsample(x)
        x = _upsample_like(x, _x)
        x = self.final(x)
        output = x

        return F.sigmoid(output),0,0,0,0,0,0
        
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()



class JITNET_SIDE(nn.Module):
    def __init__(self,in_ch=3,out_ch=1,
                 encoder_channels=[8, 32, 64, 64, 128],
                 encoder_strides=[2, 2, 2, 2, 2],
                 decoder_channels=[128, 64, 32, 32, 32],
                 decoder_strides=[1, 1, 1, 1, 1],
                 decoder_upsamples=[2, 2, 4, 1, 2],
                 side_upsamples=[32, 16, 8, 2, 2],
                 bn_momentum=0.001,
                 **_):
        super(JITNET_SIDE, self).__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(3, encoder_channels[0], 3, 2, 1),
            nn.BatchNorm2d(encoder_channels[0], momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(encoder_channels[0], encoder_channels[1], 3, 2, 1),
            nn.BatchNorm2d(encoder_channels[1], momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )

        self.enc_blocks = []
        for i in range(2, len(encoder_channels)):
            self.enc_blocks.append(basic_block(encoder_channels[i - 1],
                                               encoder_channels[i],
                                               encoder_strides[i]))
        self.enc_blocks = nn.ModuleList(self.enc_blocks)
        self.dec_blocks = []
        self.sides = []
        self.upscores = []
        prev_c = encoder_channels[-1]
        for i in range(0, len(decoder_channels) - 2):
            self.sides.append(nn.Conv2d(in_channels=prev_c,out_channels=1,kernel_size=3,padding=1))
            self.upscores.append(nn.Upsample(scale_factor=side_upsamples[i], mode='bilinear'))
            self.dec_blocks.append(basic_block(prev_c,
                                               decoder_channels[i],
                                               decoder_strides[i],
                                               decoder_upsamples[i]))
            prev_c = decoder_channels[i] + encoder_channels[-i - 2]
        self.dec_blocks = nn.ModuleList(self.dec_blocks)
        self.sides = nn.ModuleList(self.sides)
        self.upscores = nn.ModuleList(self.upscores)

        self.dec1 = nn.Sequential(
            nn.Conv2d(decoder_channels[-3], decoder_channels[-2], 3, 1, 1),
            nn.BatchNorm2d(decoder_channels[-2], momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(decoder_channels[-2], decoder_channels[-1], 3, 1, 1),
            nn.BatchNorm2d(decoder_channels[-1], momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )

        self.dec_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final = nn.Conv2d(decoder_channels[-1], 1, 1)
        self._initialize_weights()

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        down_x = []
        final_x = []
        for b in self.enc_blocks:
            x = b(x)
            down_x.append(x)
        for i, b in enumerate(self.dec_blocks):
            x_tmp = (self.sides[i])(x)
            x_tmp = (self.upscores[i])(x_tmp)
            final_x.append(x_tmp)
            x = b(x)
            if i < len(self.dec_blocks) - 1:
                dx = down_x[-i - 2]
                x = torch.cat([dx, x[:, :, :dx.shape[2], :dx.shape[3]]], dim=1)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec_upsample(x)
        x = self.final(x)
        
        final_x.append(x)
        output = torch.cat(final_x, dim=1)
        print(output.shape)
        
        return F.sigmoid(output),0,0,0,0,0,0
        
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()