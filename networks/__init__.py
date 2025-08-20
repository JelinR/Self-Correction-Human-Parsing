from __future__ import absolute_import

from networks.AugmentCE2P import resnet101, resnet101_fusion_conv, resnet101_fusion_cross, \
    resnet101_multiscale, resnet101_mod, resnet101_fusion_conv_blocks, resnet101_fusion_conv_alt, \
    resnet101_fusion_conv_alt_blocks, resnet101_multiscale_btw3

from networks.HRNet_SCHP import lite_hrnet_schp_custom, lite_hrnet_schp_30_small, \
                                    hrnet_schp_48, hrnet_schp_48_top, \
                                    hrnet_48_ocr

from networks.ResNeSt_SCHP import ResNeSt101

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

    'ResNeSt101': ResNeSt101
}


def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model arch: {}".format(name))
    return __factory[name](*args, **kwargs)