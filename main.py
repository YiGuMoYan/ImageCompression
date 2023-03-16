import copy

import cv2
import numpy
from PIL import Image

splitSize = 8


def rgb2YCbCr(rgbNumpy):
    """
    将 RGB 转化为 YCbCr
    :param rgbNumpy: RGB矩阵
    :return: YCbCr矩阵
    """
    yCbCrNumpy = copy.deepcopy(rgbNumpy)
    r = rgbNumpy[:, :, 0]
    g = rgbNumpy[:, :, 1]
    b = rgbNumpy[:, :, 2]
    yCbCrNumpy[:, :, 0] = 0.299 * r + 0.587 * g + 0.114 * b
    yCbCrNumpy[:, :, 1] = -0.1687 * r - 0.3313 * g + 0.5 * b + 128
    yCbCrNumpy[:, :, 2] = 0.5 * r - 0.4187 * g - 0.0813 * b + 128
    return numpy.uint8(yCbCrNumpy)


def yCbCr2RGB(yCbCrNumpy):
    """
    将 YCbCr 转化为 RGB
    :param yCbCrNumpy: YCbCr矩阵
    :return: RGB矩阵
    """
    rgbNumpy = copy.deepcopy(yCbCrNumpy)
    y = yCbCrNumpy[:, :, 0]
    cb = yCbCrNumpy[:, :, 1]
    cr = yCbCrNumpy[:, : 2]
    rgbNumpy[:, :, 0] = y + 1.402 * cr
    rgbNumpy[:, :, 1] = y - 0.34414 * cb - 0.71414 * cr
    rgbNumpy[:, :, 2] = y + 1.772 * cb
    return numpy.uint8(rgbNumpy)


def fullNumpy(rawNumpy):
    """
    为了给矩阵进行分割，将不足的地方补为 0
    :param rawNumpy: 原矩阵
    :return: 补全矩阵
    """
    width, height = len(rawNumpy[0]), len(rawNumpy)
    addWidth, addHeight = 0, 0
    if width % splitSize:
        addWidth = splitSize - width % splitSize
    if addHeight % splitSize:
        addHeight = splitSize - height % splitSize
    return numpy.pad(rawNumpy, ((0, addWidth), (0, addHeight)), "constant", constant_values=(0, 0))


def splitNumpy(rawNumpy):
    """
    将原矩阵分块为大小为 splitSize * splitSize 的矩阵
    :param rawNumpy: 原矩阵
    :return: 分割后的矩阵列表 每行的矩阵个数
    """
    splitList = []
    rowSize = 0
    vList = numpy.vsplit(rawNumpy, splitSize)
    for item in vList:
        hList = numpy.hsplit(item, splitList)
        splitList.append(hList)
        rowSize = len(hList)
    return splitList, rowSize


def markNumpy(rawNumpy, percent):
    """
    对分割后的矩阵进行处理
    将右下角即具有高频信号的低能量区适当简化为0
    :param rawNumpy:
    :return:
    """
    width, height = len(rawNumpy[0]), len(rawNumpy)
    resNumpy = copy.deepcopy(rawNumpy)
    for i in range(round(splitSize ** 2 * percent)):
        resNumpy[width - 1][height - 1] = 0


if __name__ == '__main__':
    url = "./g.jpg"
    rawImage = Image.open(url)
    rawNumpy = numpy.array(rawImage)
