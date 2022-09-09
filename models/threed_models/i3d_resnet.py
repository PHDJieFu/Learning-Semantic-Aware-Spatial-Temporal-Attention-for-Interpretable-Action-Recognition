import numpy as np
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from models.threed_models.fpn import FPN
from models.threed_models.none_local import NONLocalBlock3D
from models.inflate_from_2d_model import inflate_from_2d_model

__all__ = ['i3d_resnet']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def BasicConv3d(in_planes, out_planes, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0),
                bias=False):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=(1, 1, 1), padding=0, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = BasicConv3d(inplanes, planes, kernel_size=(3, 3, 3),
                                 stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = BasicConv3d(planes, planes, kernel_size=3,
                                 stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, padding=0, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = BasicConv3d(inplanes, planes, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = BasicConv3d(planes, planes, kernel_size=(3, 3, 3), stride=stride, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = BasicConv3d(planes, planes * self.expansion, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        # self.senet = nn.Sequential(
        #     nn.AdaptiveAvgPool3d(output_size=(1, 1, 1)),
        #     nn.Conv3d(planes * self.expansion, planes * self.expansion // 16, kernel_size=(1, 1, 1)),
        #     nn.ReLU(),
        #     nn.Conv3d(planes * self.expansion // 16, planes * self.expansion, kernel_size=(1, 1, 1)),
        #     nn.Sigmoid()
        # )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # SeNet module weighted output
        # se_channel_weight = self.senet(out)
        # out = out * se_channel_weight

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)

        return out


class I3D_ResNet(nn.Module):
    def __init__(self, depth, num_classes=1000, dropout=0.5, without_t_stride=False, zero_init_residual=False):
        super(I3D_ResNet, self).__init__()
        layers = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3]}[depth]
        block = BasicBlock if depth < 50 else Bottleneck
        self.depth = depth
        self.without_t_stride = without_t_stride
        self.inplanes = 64
        self.t_s = 1 if without_t_stride else 2
        self.conv1 = BasicConv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        # attention branch module between layer3 and layer4
        self.att_layer4 = self._make_layer(block, 256, layers[3])
        self.bn_att = nn.BatchNorm3d(256 * 4)
        self.att_conv11 = BasicConv3d(1024, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.att_conv12 = BasicConv3d(512, 174, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.bn_att2 = nn.BatchNorm3d(174)
        self.att_conv3 = BasicConv3d(174, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.bn_att3 = nn.BatchNorm3d(1)
        self.sigmoid = nn.Sigmoid()
        self.att_conv2 = BasicConv3d(174, 174, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False)
        self.att_conv4 = BasicConv3d(174, 174, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False)

        # senet module3
        # self.senet_module3 = nn.Sequential(
        #     nn.AdaptiveAvgPool3d(output_size=(1, 1, 1)),
        #     nn.Conv3d(1024, 1024 // 8, kernel_size=(1, 1, 1)),
        #     nn.ReLU(),
        #     nn.Conv3d(1024 // 8, 1024, kernel_size=(1, 1, 1)),
        #     nn.Sigmoid()
        # )

        # none_local module3
        # self.nl_attention_module3 = NONLocalBlock3D(1024, sub_sample=False, bn_layer=True)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # none_local module4
        # self.nl_attention_module4 = NONLocalBlock3D(2048, sub_sample=False, bn_layer=True)

        # senet module4
        # self.senet_module4 = nn.Sequential(
        #     nn.AdaptiveAvgPool3d(output_size=(1, 1, 1)),
        #     nn.Conv3d(2048, 2048 // 8, kernel_size=(1, 1, 1)),
        #     nn.ReLU(),
        #     nn.Conv3d(2048 // 8, 2048, kernel_size=(1, 1, 1)),
        #     nn.Sigmoid()
        # )

        # FPN module
        # self.fpn_module = FPN()

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def mean(self, modality='rgb'):
        return [0.485, 0.456, 0.406] if modality == 'rgb' else [0.5]

    def std(self, modality='rgb'):
        return [0.229, 0.224, 0.225] if modality == 'rgb' else [np.mean([0.229, 0.224, 0.225])]
    
    @property
    def network_name(self):
        name = 'i3d-resnet-{}'.format(self.depth)

        if not self.without_t_stride:
            name += '-ts'.format(self.depth)
        return name

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                BasicConv3d(self.inplanes, planes * block.expansion, kernel_size=(1, 1, 1),
                            stride=(self.t_s if stride == 2 else 1, stride, stride)),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride=(self.t_s if stride == 2 else 1, stride, stride),
                            padding=1, downsample=downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, padding=1))
        return nn.Sequential(*layers)
    
    def forward(self, x, target):
        x = self.conv1(x)           # 64 * 64 * 16 * 56 * 56
        num_frames_att = x.shape[2]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)          # 64 * 256 * 16 * 28 * 28
        x = self.layer2(x)          # 64 * 512 * 16 * 14 * 14
        x = self.layer3(x)          # 64 * 1024 * 16 * 7 * 7

        # attention branch between layer3 and layer4 #
        ax = self.bn_att(self.att_layer4(x))    # 64 * 1024 * 16 * 7 * 7
        ax = self.relu(self.bn_att2(self.att_conv12(self.att_conv11(ax))))        # 64 * 174 * 16 * 7 * 7

        ax_dim_re = self.bn_att3(self.att_conv3(ax))        # 64 * 1 * 16 * 7 * 7
        ax_att_t_ = F.adaptive_avg_pool3d(ax_dim_re, output_size=(num_frames_att, 1, 1))    # 64 * 1 * 16 * 1 * 1
        self.ax_att_t = self.sigmoid(ax_att_t_)                     # 64 * 1 * 16 * 1 * 1
        self.ax_att_xy = self.sigmoid(ax_dim_re)                 # 64 * 1 * 16 * 7 * 7

        ax = self.att_conv2(ax)                                         # 64 * 174 * 16 * 7 * 7
        ax = F.adaptive_avg_pool3d(ax, output_size=(1, 1, 1))           # 64 * 174 * 1 * 1 * 1
        ax = ax.view(ax.size(0), -1)

        # ax = ax * ax_cls_                    # 64 * 174 * 16 * 7 * 7 (channel weighted feature map)

        ax_att = self.ax_att_t * self.ax_att_xy         # 64 * 1 * 16 * 7 * 7
        rx = x * ax_att                                 # 64 * 1024 * 16 * 7 * 7
        rx = x + rx                                     # 64 * 1024 * 16 * 7 * 7

        # rx_weight3 = self.senet_module3(rx)               # 64 * 1024 * 1 * 1 * 1
        # rx = rx_weight3 * rx                             # 64 * 1024 * 16 * 7 * 7
        # ########################################################################################################################## #

        # rx = self.nl_attention_module(rx)
        rx = self.layer4(rx)                        # 64 * 2048 * 16 * 4 * 4
        # rx_weight4 = self.senet_module4(rx)         # 64 * 2048 * 1 * 1 * 1
        # rx = rx_weight4 * rx                        # 64 * 2048 * 16 * 4 * 4

        # fpn module for layer3 and layer4
        # fpn_input_features = [x, rx]
        # rx = rx + F.interpolate(self.fpn_module(fpn_input_features), size=(16, 4, 4), mode='nearest')               # 64 * 2048 * 16 * 4 * 4

        num_frames = rx.shape[2]     # 16
        rx = F.adaptive_avg_pool3d(rx, output_size=(num_frames, 1, 1))
        # N x 1024 x ((F/8)-1) x 1 x 1
        rx = rx.squeeze(-1)
        rx = rx.squeeze(-1)
        rx = rx.transpose(1, 2)
        n, c, nf = rx.size()
        rx = rx.contiguous().view(n * c, -1)
        rx = self.dropout(rx)
        rx = self.fc(rx)
        rx = rx.view(n, c, -1)
        # N x num_classes x ((F/8)-1)
        rx = torch.mean(rx, 1)

        return ax, rx, self.ax_att_t, self.ax_att_xy


def i3d_resnet(depth, num_classes, dropout, without_t_stride, **kwargs):
    model = I3D_ResNet(depth, num_classes=num_classes, dropout=dropout, without_t_stride=without_t_stride)

    new_model_state_dict = model.state_dict()
    state_dict = model_zoo.load_url(model_urls['resnet{}'.format(depth)], map_location='cpu', progress=True)
    state_d = inflate_from_2d_model(state_dict, new_model_state_dict, skipped_keys=['fc'])
    model.load_state_dict(state_d, strict=False)
    return model


if __name__ == '__main__':
    from torchsummary import torchsummary
    model = i3d_resnet(50, 400, 0.5, without_t_stride=False)

    dummy_data = (3, 64, 224, 224)
    model.eval()
    model_summary = torchsummary.summary(model, input_size=dummy_data)
    print(model_summary)
