# -*- coding: utf-8 -*-
# @Author : youngx
# @Time : 9:26  2022-10-19

# import paddlehub as hub
# # import cv2
import numpy as np
from PIL import Image
import os
import math
import paddle
# paddle.enable_static()


# net = paddle.vision.models.resnet50(pretrained=True)
# print(net)
# net.eval()
#
# net = paddle.jit.to_static(net)  # 动静转换
# x = paddle.rand([1, 3, 224, 224])
# y = net(x)
# print(y.shape)
# path = "./resnet50"
# paddle.jit.save(net, path)

DATA_DIM = 224
img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))


def load_label_info(file_path):
    with open(file_path, 'r') as fr:
        return fr.read().split("\n")[:-1]


def resize_short(img, target_size):
    percent = float(target_size) / min(img.size[0], img.size[1])
    resized_width = int(round(img.size[0] * percent))
    resized_height = int(round(img.size[1] * percent))
    img = img.resize((resized_width, resized_height), Image.LANCZOS)
    return img


def crop_image(img, target_size, center):
    width, height = img.size
    size = target_size
    if center == True:
        w_start = (width - size) / 2
        h_start = (height - size) / 2
    else:
        w_start = np.random.randint(0, width - size + 1)
        h_start = np.random.randint(0, height - size + 1)
    w_end = w_start + size
    h_end = h_start + size
    img = img.crop((w_start, h_start, w_end, h_end))
    return img


def process_image(img):
    img = resize_short(img, target_size=256)
    img = crop_image(img, target_size=DATA_DIM, center=True)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img).astype('float32').transpose((2, 0, 1)) / 255
    img -= img_mean
    img /= img_std
    return img


def test_reader(paths=None, images=None):
    """data generator
    :param paths: path to images.
    :type paths: list, each element is a str
    :param images: data of images, [N, H, W, C]
    :type images: numpy.ndarray
    """
    img_list = []
    if paths:
        for img_path in paths:
            assert os.path.isfile(img_path), "The {} isn't a valid file path.".format(img_path)
            img = Image.open(img_path)
            # img = cv2.imread(img_path)
            img_list.append(img)
    if images is not None:
        for img in images:
            img_list.append(Image.fromarray(np.uint8(img)))

    process = []
    for im in img_list:
        im = process_image(im)
        process.append(im)
    return process


if __name__ == '__main__':
    net = paddle.vision.models.resnet50(pretrained=True)
    print(net)
    net.eval()

    net = paddle.jit.to_static(net)  # 动静转换
    x = paddle.rand([1, 3, 224, 224])
    y = net(x)
    print(y.shape)
    path = "./resnet50"
    paddle.jit.save(net, path)

    load_func = paddle.jit.load(path)
    load_result = load_func(x)


    labels = load_label_info("./label_file.txt")
    dataList = test_reader([r"E:\Share_Files\compareAI\paddel_demo\test_img_cat.jpg",
                        r"E:\Share_Files\compareAI\paddel_demo\test_img_dog.jpg",
                        r"E:\Share_Files\compareAI\paddel_demo\flower.jpg"])

    for data in dataList:
        batch_data = np.array(data[None,...]).astype('float32')
        data_tensor = paddle.Tensor(batch_data)   # . PaddleTensor(batch_data.copy())
        origin = load_func(data_tensor)
        origin = np.array(origin)
        pred_label = np.argmax(origin)

        print(labels[pred_label])
