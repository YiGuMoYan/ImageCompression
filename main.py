import copy
import os

import cv2
import numpy
import uuid
from PIL import Image

import gradio
import matplotlib.pyplot as plt


def RGB2YUV(r, g, b):
    """
    RGB 通道改为 YUV 通道
    :param r: R
    :param g: G
    :param b: B
    :return: YUV
    """
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.1687 * r - 0.3313 * g + 0.5 * b + 128
    v = 0.5 * r - 0.419 * g - 0.081 * b + 128
    return y, u, v


def YUV2RGB(y, u, v):
    """
    YUV 通道改为 RGB 通道
    :param y: Y
    :param u: U
    :param v: V
    :return:
    """
    r = y + 1.402 * (v - 128)
    g = y - 0.34414 * (u - 128) - 0.71414 * (v - 128)
    b = y + 1.772 * (u - 128)
    return r, g, b


def quantizationY(raw, quality):
    """
    Y量化计算
    :param raw: 原矩阵
    :param quality: 质量
    :return:
    """
    qY = numpy.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ])
    qY = setQuantization(qY, quality)
    return numpy.round(raw / qY) * qY


def quantizationUV(raw, quality):
    """
    UV量化计算
    :param raw: 原矩阵
    :param quality: 质量
    :return:
    """
    qUV = numpy.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ])
    qUV = setQuantization(qUV, quality)
    return numpy.round(raw / qUV) * qUV


def setQuantization(q, quality):
    """
    自定义量化矩阵
    :param q: 原来量化矩阵
    :param quality: 质量
    :return:
    """
    return q * quality


def zigzag(raw):
    width, height = len(raw[0]), len(raw)
    x, y, flag = 0, 0, 0
    res = []
    while x != width - 1 or y != height - 1:
        res.append(raw[y, x])
        # x + y 为偶数，向右或右上移动
        if (x + y) % 2 == 0:
            if ((y == 0 and not x == width - 1) or (y == height - 1 and not x == 0)) and not flag:
                x += 1
                flag = 0
            elif x == width - 1:
                y += 1
                flag = 1
            else:
                x = x + 1
                y = y - 1
                flag = 0
        # x + y 为奇数，向下或左下移动
        else:
            if ((x == 0 and not y == height - 1) or (x == width - 1 and not y == 0)) and not flag:
                y += 1
                flag = 0
            elif y == height - 1:
                x += 1
                flag = 1
            else:
                flag = 0
                x = x - 1
                y = y + 1
    res.append(raw[y, x])
    return res


class ImageCompression:
    def __init__(self, raw, quality):
        # 原数组
        self.__rawNumpy = raw
        # 图片宽度 高度 色彩通道
        self.__height, self.__width = -1, -1
        self.__channel = 1 if raw.ndim == 2 else 3
        if self.__channel == 3:
            self.__height, self.__width, self.__channel = raw.shape
        elif self.__channel == 1:
            self.__height, self.__width = raw.shape
        self.__addWidth, self.__addHeight = -1, -1
        self.__quality = quality
        self.resImage = ''

    def fillImage(self, raw):
        """
        图像填充
        色彩取样时 1/2
        矩阵分块时 1/8
        :return:
        """
        resNumpy = copy.deepcopy(raw)
        if self.__addWidth == -1 and self.__addHeight == -1:
            if self.__width % 16 != 0:
                self.__addWidth = 16 - self.__width % 16
            if self.__height % 16 != 0:
                self.__addHeight = 16 - self.__height % 16
        resNumpy = numpy.pad(raw, ((0, 0 if self.__addHeight == -1 else self.__addHeight),
                                   (0, 0 if self.__addWidth == -1 else self.__addWidth)), "constant")
        self.__height, self.__width = resNumpy.shape
        return resNumpy

    def encode(self, raw, flag):
        """
        图像压缩
        :return:
        """
        fillNumpy = self.fillImage(raw)
        shape = (self.__height // 8, self.__width // 8, 8, 8)
        strides = fillNumpy.itemsize * numpy.array([self.__width * 8, 8, self.__width, 1])
        blockNumpy = numpy.lib.stride_tricks.as_strided(fillNumpy, shape=shape, strides=strides)
        res = []
        for i in range(self.__height // 8):
            for j in range(self.__width // 8):
                if flag == 0:
                    resDct = cv2.dct(blockNumpy[i, j].astype('float'))
                    resQuantization = quantizationY(resDct, self.__quality)
                    resIdct = cv2.idct(resQuantization)
                    res.append(resIdct)
                elif flag == 1:
                    resDct = cv2.dct(blockNumpy[i, j].astype('float'))
                    resQuantization = quantizationUV(resDct, self.__quality)
                    resIdct = cv2.idct(resQuantization)
                    res.append(resIdct)
        return res

    def mergeNumpy(self, splitList):
        """
        拼接矩阵
        :param splitList: 分割后的矩阵数组
        :return: 合成的矩阵
        """
        size = self.__width // 8
        rowNumpyList = []
        for i in range(0, len(splitList), size):
            rowNumpy = numpy.concatenate(tuple(splitList[i: i + size]), axis=1)
            rowNumpyList.append(rowNumpy)
        resNumpy = numpy.concatenate(tuple(rowNumpyList), axis=0)
        return resNumpy

    def splitNumpy(self, raw):
        """
        切割图形，将图形复原
        :param raw: 切割前矩阵
        :return: 切割后矩阵
        """
        resNumpy = copy.deepcopy(raw)
        if self.__addWidth != -1:
            resNumpy = numpy.split(resNumpy, [self.__width - self.__addWidth], axis=1)[0]
        if self.__addHeight != -1:
            resNumpy = numpy.split(resNumpy, [self.__height - self.__addHeight])[0]
        return resNumpy

    def compress(self):
        """
        图像压缩
        :return:
        """
        if self.__channel == 3:
            r, g, b = self.__rawNumpy[:, :, 0], self.__rawNumpy[:, :, 1], self.__rawNumpy[:, :, 2]
            y, u, v = RGB2YUV(r, g, b)
            yList = self.encode(y, 0)
            uList = self.encode(u, 1)
            vList = self.encode(v, 1)
            y = self.mergeNumpy(yList)
            u = self.mergeNumpy(uList)
            v = self.mergeNumpy(vList)
            r, g, b = YUV2RGB(y, u, v)
            r = Image.fromarray(self.splitNumpy(r)).convert('L')
            g = Image.fromarray(self.splitNumpy(g)).convert('L')
            b = Image.fromarray(self.splitNumpy(b)).convert('L')
            self.resImage = Image.merge("RGB", (r, g, b))
            # resImage.save("1-1.jpg")
        elif self.__channel == 1:
            y = self.__rawNumpy[:, :, ]
            yList = self.encode(y, 0)
            y = self.mergeNumpy(yList)
            self.resImage = Image.fromarray(y).convert("L")
            # resImage.save("1-1.jpg")


def main(image, quality, customization):
    """
    主函数
    :param image: 原图像
    :param quality: 提供的压缩比例
    :param customization: 自定义压缩比例
    :return: 压缩后的图像 对比图 压缩比
    """

    # 创建对应结果文件夹
    name = str(uuid.uuid1())
    os.mkdir("./resImage/" + name)

    # 获取原图像大小
    Image.fromarray(image).save("./resImage/" + name + "/raw.jpg")
    rawSize = os.stat("./resImage/" + name + "/raw.jpg")

    # 显示原图像
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("raw - " + str(rawSize.st_size))
    plt.xticks([]), plt.yticks([]), plt.axis('off')

    # 解析原图像
    rawNumpy = numpy.array(image)

    # 图像压缩
    image = ImageCompression(rawNumpy, quality if quality > customization else customization)
    image.compress()

    # 保存结果图像
    image.resImage.save("./resImage/" + name + "/res.jpg")
    resSize = os.stat("./resImage/" + name + "/res.jpg")

    # 显示压缩后的图像
    plt.subplot(1, 2, 2)
    plt.imshow(image.resImage)
    plt.title("res - " + str(resSize.st_size))
    plt.xticks([]), plt.yticks([]), plt.axis('off')

    # 保存对比图
    f = plt.gcf()
    f.savefig("./resImage/" + name + "/plt.jpg", dpi=600)
    f.clear()

    return image.resImage, Image.open("./resImage/" + name + "/plt.jpg"), "compression ratio = " + str(
        resSize.st_size / rawSize.st_size)


if __name__ == "__main__":
    gr = gradio.Interface(fn=main, inputs=[gradio.Image(label="raw"), gradio.Slider(0, 5), gradio.Number()],
                          outputs=[gradio.Image(label="result"), gradio.Image(label="plot"), gradio.Text(label="compression ratio")])
    gr.title = "ImageCompression - JPEG"
    gr.launch()
