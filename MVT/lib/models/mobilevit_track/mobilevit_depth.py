#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from torch import nn
import argparse
from typing import Dict, Tuple, Optional

from .cvnets_utils import logger

# from . import register_cls_models
from .config.mobilevit import get_configuration
from .layers import ConvLayer
from .modules import InvertedResidual, MobileViT_Track_Depth_Block
from .base_backbone import BaseEncoder


# @register_cls_models("mobilevit")
class MobileViTDepth(BaseEncoder):
    """
    This class implements the `MobileViT architecture <https://arxiv.org/abs/2110.02178?context=cs.LG>`_
    """

    def __init__(self, opts, training=False, *args, **kwargs) -> None:

        image_channels = 3
        out_channels = 16

        mobilevit_config = get_configuration(opts=opts)

        # super().__init__(opts, *args, **kwargs)
        super().__init__(training=training)

        # store model configuration in a dictionary
        self.model_conf_dict = dict()

        # conv_1 (i.e., the first conv3x3 layer) output for
        self.conv_1 = ConvLayer(
            opts=opts,
            in_channels=image_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            use_norm=True,
            use_act=True,
        )
        self.conv_1_depth = ConvLayer(
            opts=opts,
            in_channels=image_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            use_norm=True,
            use_act=True,
        )

        self.model_conf_dict["conv1"] = {"in": image_channels, "out": out_channels}

        # layer_1 (i.e., MobileNetV2 block) output FROZEN
        # layer_1_depth (i.e., MobileNetV2 block) output
        in_channels = out_channels
        self.layer_1, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer1"]
        )
        self.layer_1_depth, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer1"]
        )
        self.model_conf_dict["layer1"] = {"in": in_channels, "out": out_channels}

        # layer_2 (i.e., MobileNetV2 with down-sampling + 2 x MobileNetV2) output FROZEN
        # layer_2_depth (i.e., MobileNetV2 with down-sampling + 2 x MobileNetV2) output
        in_channels = out_channels
        self.layer_2_depth, _ = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer2"]
        )
        self.layer_2, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer2"]
        )
        self.model_conf_dict["layer2"] = {"in": in_channels, "out": out_channels}

        # depth tokens to input to layer_3
        self.prompt_3_depth, _ = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["prompt3"]
        )

        # layer_3 (i.e., MobileNetV2 with down-sampling + 2 x MobileViT-Track block) output
        # layer_3_depth (i.e., MobileNetV2 with down-sampling + 2 x MobileViT-Track block) output
        in_channels = out_channels
        self.layer_3_depth, _ = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer3"]
        )
        self.layer_3, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer3"]
        )
        self.model_conf_dict["layer3"] = {"in": in_channels, "out": out_channels}
        
        # depth tokens to input to layer_3
        self.prompt_4_depth, _ = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["prompt4"]
        )

        # layer_4 (i.e., MobileNetV2 with down-sampling + 4 x MobileViT-Track block) output
        in_channels = out_channels
        self.layer_4, out_channels = self._make_layer(
            opts=opts,
            input_channel=in_channels,
            cfg=mobilevit_config["layer4"],
            dilate=False,
        )
        self.model_conf_dict["layer4"] = {"in": in_channels, "out": out_channels}

        # check model
        # self.check_model()

        # weight initialization
        self.reset_parameters(opts=opts)

    def _make_layer(
        self,
        opts,
        input_channel,
        cfg: Dict,
        dilate: Optional[bool] = False,
        *args,
        **kwargs
        ) -> Tuple[nn.Sequential, int]:
        block_type = cfg.get("block_type", "mobilevit")
        if block_type.lower() == "mobilevit":
            return self._make_mit_layer(
                opts=opts, input_channel=input_channel, cfg=cfg, dilate=dilate
            )
        else:
            return self._make_mobilenet_layer(
                opts=opts, input_channel=input_channel, cfg=cfg
            )

    @staticmethod
    def _make_mobilenet_layer(
        opts, input_channel: int, cfg: Dict, *args, **kwargs
    ) -> Tuple[nn.Sequential, int]:
        output_channels = cfg.get("out_channels")
        num_blocks = cfg.get("num_blocks", 2)
        expand_ratio = cfg.get("expand_ratio", 4)
        block = []

        for i in range(num_blocks):
            stride = cfg.get("stride", 1) if i == 0 else 1

            layer = InvertedResidual(
                opts=opts,
                in_channels=input_channel,
                out_channels=output_channels,
                stride=stride,
                expand_ratio=expand_ratio,
            )
            block.append(layer)
            input_channel = output_channels
        return nn.Sequential(*block), input_channel

    def _make_mit_layer(
        self,
        opts,
        input_channel,
        cfg: Dict,
        dilate: Optional[bool] = False,
        *args,
        **kwargs
    ) -> Tuple[nn.Sequential, int]:
        prev_dilation = self.dilation
        block = []
        stride = cfg.get("stride", 1)

        if stride == 2:
            if dilate:
                self.dilation *= 2
                stride = 1

            layer = InvertedResidual(
                opts=opts,
                in_channels=input_channel,
                out_channels=cfg.get("out_channels"),
                stride=stride,
                expand_ratio=cfg.get("mv_expand_ratio", 4),
                dilation=prev_dilation,
            )

            block.append(layer)
            input_channel = cfg.get("out_channels")

        head_dim = cfg.get("head_dim", 32)
        transformer_dim = cfg["transformer_channels"]
        ffn_dim = cfg.get("ffn_dim")
        if head_dim is None:
            num_heads = cfg.get("num_heads", 4)
            if num_heads is None:
                num_heads = 4
            head_dim = transformer_dim // num_heads

        if transformer_dim % head_dim != 0:
            logger.error(
                "Transformer input dimension should be divisible by head dimension. "
                "Got {} and {}.".format(transformer_dim, head_dim)
            )

        block.append(
            MobileViT_Track_Depth_Block(
                opts=opts,
                in_channels=input_channel,
                transformer_dim=transformer_dim,
                ffn_dim=ffn_dim,
                n_transformer_blocks=cfg.get("transformer_blocks", 1),
                patch_h=cfg.get("patch_h", 2),
                patch_w=cfg.get("patch_w", 2),
                dropout=getattr(opts, "model.classification.mit.dropout", 0.1),
                ffn_dropout=getattr(opts, "model.classification.mit.ffn_dropout", 0.0),
                attn_dropout=getattr(
                    opts, "model.classification.mit.attn_dropout", 0.1
                ),
                head_dim=head_dim,
                no_fusion=getattr(
                    opts,
                    "model.classification.mit.no_fuse_local_global_features",
                    False,
                ),
                conv_ksize=getattr(
                    opts, "model.classification.mit.conv_kernel_size", 3
                ),
            )
        )

        return nn.Sequential(*block), input_channel
