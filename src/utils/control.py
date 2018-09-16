# -*- coding: utf-8 -*-
"""
@Author: xiezizhe
@Date: 2018/9/16 下午5:36
"""

import sys

VERSION = sys.version.split(" ")[0]


def high_version(version=VERSION, flag=False):
    if flag:
        print("python version: " + version)
    if version[0] == "3":
        return True
    if version[0] == "2":
        return False

    raise RuntimeError('unkown version')
