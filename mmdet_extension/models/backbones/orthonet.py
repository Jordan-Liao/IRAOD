import torch
import torch.nn as nn
from mmcv.runner import BaseModule, load_checkpoint
from mmdet.models.builder import BACKBONES
from mmdet.utils import get_root_logger


class OrthoChannelAttention(nn.Module):
    """Orthogonal Channel Attention module.
    Uses QR decomposition for orthogonal projection and
    bottleneck FC layers (like SE-Net) for channel attention.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )
        # learnable orthogonal basis
        self.ortho_weight = nn.Parameter(torch.empty(channels, channels))
        nn.init.orthogonal_(self.ortho_weight)

    def forward(self, x):
        b, c, _, _ = x.size()
        # channel descriptor
        y = self.avg_pool(x).view(b, c)
        # orthogonal projection
        q, _ = torch.linalg.qr(self.ortho_weight)
        y = y @ q
        # bottleneck attention
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, 3, stride=stride,
                     padding=dilation, dilation=dilation, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 reduction=16, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.attn = OrthoChannelAttention(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.attn(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 reduction=16, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.attn = OrthoChannelAttention(planes * self.expansion, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.attn(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


_BLOCK_MAP = {
    18: (BasicBlock, [2, 2, 2, 2]),
    34: (BasicBlock, [3, 4, 6, 3]),
    50: (Bottleneck, [3, 4, 6, 3]),
    101: (Bottleneck, [3, 4, 23, 3]),
    152: (Bottleneck, [3, 8, 36, 3]),
}


@BACKBONES.register_module()
class OrthoNet(BaseModule):
    """ResNet backbone with Orthogonal Channel Attention.

    Args:
        depth (int): Network depth (18, 34, 50, 101, 152).
        reduction (int): Channel reduction ratio for attention. Default: 16.
        in_channels (int): Input image channels. Default: 3.
        num_stages (int): Number of stages (1-4). Default: 4.
        out_indices (tuple): Output from which stages. Default: (0,1,2,3).
        frozen_stages (int): Stages to freeze (-1 means none). Default: -1.
        norm_eval (bool): Keep BN in eval mode. Default: True.
        style (str): Unused, kept for config compat. Default: 'pytorch'.
        init_cfg (dict): Initialization config. Default: None.
    """

    def __init__(self, depth=50, reduction=16, in_channels=3, num_stages=4,
                 out_indices=(0, 1, 2, 3), frozen_stages=-1, norm_eval=True,
                 style='pytorch', init_cfg=None, **kwargs):
        super().__init__(init_cfg=init_cfg)
        assert depth in _BLOCK_MAP, f'Unsupported depth {depth}'
        block, layers = _BLOCK_MAP[depth]
        self.depth = depth
        self.reduction = reduction
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.inplanes = 64

        norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        stage_planes = [64, 128, 256, 512]
        self.res_layers = []
        for i in range(num_stages):
            stride = 1 if i == 0 else 2
            layer = self._make_layer(block, stage_planes[i], layers[i],
                                     stride=stride, reduction=reduction,
                                     norm_layer=norm_layer)
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

    def _make_layer(self, block, planes, blocks, stride=1, reduction=16,
                    norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample,
                        reduction, norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                reduction=reduction, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.bn1.eval()
            for m in [self.conv1, self.bn1]:
                for param in m.parameters():
                    param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self, f'layer{i}')
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                            nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
