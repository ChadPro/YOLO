# -- coding: utf-8 --
# Copyright 2018 The LongYan. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division

from nets import base_net


net_map = {
    'base_net' : base_net
    }


def get_network(name):
    if name not in net_map:
        raise ValueError('Name of net unknown %s' % name)
    return net_map[name]