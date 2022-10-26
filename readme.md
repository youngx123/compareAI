训练框架使用 PaddlePaddle,部署框架使用 paddle-lite时，模型推理速度会不会更快？

即模型部署时的推理速度，是由`训练框架 + 部署框架` 一起产生还是单独的`部署框架产生`，

__如百度自家的推理框架paddle-lite会不会对自家的训练框架paddlepaddle进行优化__

针对以上疑问，训练框架使用`pytorch和paddlepaddle`, 部署框架使用 `MNN、paddle-lite`,分别

下载`Resnet50、VGG16`的预训练模型进行测试。

文件目录为：
```python
├─eval_img
├─get_pretrainedModels
│  ├─paddel_demo
│  │  └─vgg16_imagenet
│  └─torch_demo
│      ├─mnn_model_fp32
│      └─onnx2paddle_model
└─src
    ├─mnn_demo
    └─paddle_demo
```
`src` 为部署的c++代码， 对应的lib库下载地址为：

`get_pretrainedModels` 为下载预训练模型，并进行转换保存。

`eval_img` 简单的测试图像。

__`测试平台为 NXP 8QM`__

测试结果为：

[训练框架和推理框架对比](https://github.com/youngx123/compareAI/blob/main/doc/%E8%AE%AD%E7%BB%83%E6%A1%86%E6%9E%B6%E5%92%8C%E6%8E%A8%E7%90%86%E6%A1%86%E6%9E%B6%E5%AF%B9%E6%AF%94.pdf)