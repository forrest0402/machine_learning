# -*- coding: utf-8 -*-
import os
import sys

import matplotlib.pyplot as plt

import control

VERSION = sys.version.split(" ")[0]
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

if __name__ == '__main__':
    file_name = ROOT_PATH + "/loss_dssm_l1/loss.txt"
    x = []
    if control.high_version():
        with open(file_name, 'r', encoding="utf-8") as fr:
            for line in fr.readlines():
                x.append(float(line.replace("\n", "")))
    else:
        with open(file_name, 'r') as fr:
            for line in fr.readlines():
                if "epoch" in line and "loss" in line and "accuracy" in line:
                    x.append(float(line.split(",")[2].split(" ")[2]))

    fig, ax = plt.subplots()
    line1, = ax.plot(range(len(x)), x, '-', linewidth=2, label='loss')
    ax.legend(loc='upper right')
    plt.show()
