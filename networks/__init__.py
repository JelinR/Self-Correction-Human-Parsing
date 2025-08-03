from __future__ import absolute_import

from networks.AugmentCE2P import resnet101, resnet101_fusion_conv

__factory = {
    'resnet101': resnet101,
    'resnet101_fusion_conv': resnet101_fusion_conv
}


def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model arch: {}".format(name))
    return __factory[name](*args, **kwargs)