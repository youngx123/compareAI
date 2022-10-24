# -*- coding: utf-8 -*-
# @Author : youngx
# @Time : 19:16  2022-10-18

import paddle.nn as nn
from paddle import ParamAttr
from paddlehub.module.cv_module import ImageClassifierModule
from paddle.utils.download import get_weights_path_from_url
from paddle.nn import Conv2D, BatchNorm, Linear, Dropout
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D,  AvgPool2D
from paddle.nn.initializer import Uniform
from paddlehub.module.module import moduleinfo
from paddlehub.module.cv_module import ImageClassifierModule
import paddle
# paddle.enable_static()
__all__ = []

model_urls = {
    'vgg16': ('https://paddle-hapi.bj.bcebos.com/models/vgg16.pdparams',
              '89bbffc0f87d260be9b8cdc169c991c4'),

}

"""VGG model from
Args:
    features (nn.Layer): Vgg features create by function make_layers.
    num_classes (int): Output dim of last fc layer. If num_classes <=0, last fc layer
                        will not be defined. Default: 1000.
    with_pool (bool): Use pool before the last three fc layer or not. Default: True.
Examples:
    .. code-block:: python
        from paddle.vision.models import VGG
        from paddle.vision.models.vgg import make_layers
        vgg11_cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        features = make_layers(vgg11_cfg)
        vgg11 = VGG(features)
"""
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2D(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2D(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2D(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.Sequential(*layers)

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B':
        [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512,
        512, 512, 'M'
    ],
    'E': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512,
        'M', 512, 512, 512, 512, 'M'
    ],
}

@moduleinfo(
    name="vgg16_imagenet",
    type="CV/classification",
    author="paddlepaddle",
    author_email="",
    summary="vgg16_imagenet is a classification model, "
    "this module is trained with Baidu open sourced dataset.",
    version="1.1.0",
    meta=ImageClassifierModule)

class VGG(nn.Layer):
    def __init__(self, features, num_classes=1000, with_pool=True):
        super(VGG, self).__init__()
        self.features = features
        self.num_classes = num_classes
        self.with_pool = with_pool
        if with_pool:
            self.avgpool = nn.AdaptiveAvgPool2D((7, 7))

        if num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(4096, num_classes), )

        weight_path = get_weights_path_from_url(model_urls["vgg16"][0],
                                                model_urls["vgg16"][1])

        param = paddle.load(weight_path)
        self.set_dict(param)
        print("load prtrained model ")

    @paddle.jit.to_static
    def forward(self, x):
        x = self.features(x)

        if self.with_pool:
            x = self.avgpool(x)

        if self.num_classes > 0:
            x = paddle.flatten(x, 1)
            x = self.classifier(x)

        return x

def vgg16(pretrained=False, batch_norm=False, **kwargs):
    """VGG 16-layer model

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet. Default: False.
        batch_norm (bool): If True, returns a model with batch_norm layer. Default: False.

    Examples:
        .. code-block:: python

            from paddle.vision.models import vgg16

            # build model
            model = vgg16()

            # build vgg16 model with batch_norm
            model = vgg16(batch_norm=True)
    """
    model_name = 'vgg16'
    if batch_norm:
        model_name += ('_bn')
    arch =model_name
    cfg = 'D'
    batch_norm = batch_norm
    pretrained = pretrained

    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm))

    # if pretrained:
    #     assert arch in model_urls, "{} model do not have a pretrained model now, you should set pretrained=False".format(
    #         arch)
    #     weight_path = get_weights_path_from_url(model_urls[arch][0],
    #                                             model_urls[arch][1])
    #     param = paddle.load(weight_path)
    #     a = model.state_dict()
    #     model.load_dict(param)
    #     print("load prtrained model ")

    ### save model
    print(model)
    model.eval()

    # vgg16 = paddle.jit.to_static(vgg16)  # 动静转换
    x = paddle.rand([1, 3, 224, 224])
    origin = model(x)
    # net = paddle.jit.to_static(net)
    path = "./vgg16"
    paddle.jit.save(model, path)

    load_func = paddle.jit.load(path)
    load_result = load_func(x)

    print(origin - load_result)


if __name__ == '__main__':
    vgg16(pretrained=True)
    # return model
    # return _vgg(model_name, 'D', batch_norm, pretrained, **kwargs)