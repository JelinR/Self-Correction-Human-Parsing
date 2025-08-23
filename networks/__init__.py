from __future__ import absolute_import

from networks.AugmentCE2P import resnet101, resnet101_fusion_conv, resnet101_fusion_cross, \
    resnet101_multiscale, resnet101_mod, resnet101_fusion_conv_blocks, resnet101_fusion_conv_alt, \
    resnet101_fusion_conv_alt_blocks, resnet101_multiscale_btw3

from networks.HRNet_SCHP import lite_hrnet_schp_custom, lite_hrnet_schp_30_small, \
                                    hrnet_schp_48, hrnet_schp_48_top, \
                                    hrnet_48_ocr

from networks.ResNeSt_SCHP import ResNeSt101, ResNeSt101_dilate_2, ResNeSt101_dilate_1

from networks.SegFormer_SCHP import segformer_mit_b4, segformer_b4_ce2p, segformer_b4_mixed

__factory = {
    'resnet101': resnet101,
    'resnet101_fusion_conv': resnet101_fusion_conv,
    'resnet101_fusion_cross': resnet101_fusion_cross,
    'resnet101_multiscale': resnet101_multiscale,
    'resnet101_mod': resnet101_mod,
    'resnet101_fusion_conv_blocks': resnet101_fusion_conv_blocks,
    'resnet101_fusion_conv_alt': resnet101_fusion_conv_alt,
    'resnet101_fusion_conv_alt_blocks': resnet101_fusion_conv_alt_blocks,
    'resnet101_multiscale_btw3': resnet101_multiscale_btw3,

    'lite_hrnet_schp_custom': lite_hrnet_schp_custom,
    'lite_hrnet_schp_30_small': lite_hrnet_schp_30_small,

    'hrnet_schp_48': hrnet_schp_48,
    'hrnet_schp_48_top': hrnet_schp_48_top,
    'hrnet_48_ocr': hrnet_48_ocr,

    'ResNeSt101': ResNeSt101,                           #OS = 8
    'ResNeSt101_dilate_2': ResNeSt101_dilate_2,         #OS = 16
    'ResNeSt101_dilate_1': ResNeSt101_dilate_1,         #OS = 32

    'segformer_mit_b4': segformer_mit_b4,
    'segformer_b4_ce2p': segformer_b4_ce2p,
    'segformer_b4_mixed': segformer_b4_mixed
}


def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model arch: {}".format(name))
    return __factory[name](*args, **kwargs)