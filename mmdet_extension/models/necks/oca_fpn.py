import torch.nn as nn

from mmdet.models.builder import NECKS
from mmdet.models.necks import FPN

from mmdet_extension.models.backbones.orthonet import OrthoChannelAttention


@NECKS.register_module()
class OCAFPN(FPN):
    """FPN with OrthoChannelAttention on each output level."""

    def __init__(self, *args, reduction=16, **kwargs):
        super().__init__(*args, **kwargs)
        self.oca = nn.ModuleList(
            [OrthoChannelAttention(self.out_channels, reduction) for _ in range(self.num_outs)]
        )

    def forward(self, inputs):
        outs = super().forward(inputs)
        return tuple(attn(x) for attn, x in zip(self.oca, outs))
