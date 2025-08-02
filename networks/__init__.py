from __future__ import absolute_import

from networks.AugmentCE2P import resnet101, resnet101_mod

__factory = {
    'resnet101': resnet101,
    'resnet101_mod': resnet101_mod
}


def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model arch: {}".format(name))
    return __factory[name](*args, **kwargs)