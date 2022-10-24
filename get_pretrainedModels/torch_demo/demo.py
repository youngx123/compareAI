# -*- coding: utf-8 -*-
# @Author : youngx
# @Time : 10:20  2022-10-19

import torch
import torchvision
import onnx
import onnxsim


def convertAndSimplify(model, name):
    model.eval()
    im = torch.zeros(1, 3, 224, 224)
    y = model(im)
    print(y.shape)
    torch.onnx.export(
        model,
        im,
        name,
        training=torch.onnx.TrainingMode.EVAL,
        keep_initializers_as_inputs=True,
        do_constant_folding=True,
        opset_version=12,
        input_names=["images"],
        output_names=["output"],
    )
    onnx_model = onnx.load(name)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model
    print("adf")
    output_path2 = name.replace(".onnx", "_sim.onnx")
    model_sim, flag = onnxsim.simplify(onnx_model)
    print("adf")
    if flag:
        onnx.save(model_sim, output_path2)
        # logger.log("simplify onnx successfully")


if __name__ == '__main__':
    resnet50 = torchvision.models.vgg16(pretrained=True)
    name = "torch_vgg16.onnx"
    convertAndSimplify(resnet50, name)
