# PyTorch C++ 部署例子

Pytorch 官网上有在 C++ 端加载 PyTorch 模型并进行推理的例子。
[Loading a PyTorch Model in C++](https://pytorch.org/tutorials/advanced/cpp_export.html)  
这个例子比较简单，模型是单输入单输出的。  
本项目主要给出在模型是多输入或者多输出的情况下如何处理的例子。 

# 多输出 
[export/resnet.py](export/resnet.py) 中定义的 ```ResNetConvFeatures``` 是一个 ResNet 的变种网络，
它输出 ResNet 5 个 stage 中每个 stage 最后一个卷积层的特征，一共是 5 个 tensor。  
为了支持 C++ 推理，Python 代码中这 5 个 tensor 要以 tuple 的形式组合起来。
见 ```ResNetConvFeatures``` 类 ```forward``` 函数的返回值。 
```
    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        feat0 = x

        x = self.layer1(x)
        feat1 = x
        x = self.layer2(x)
        feat2 = x
        x = self.layer3(x)
        feat3 = x
        x = self.layer4(x)
        feat4 = x

        return (feat0, feat1, feat2, feat3, feat4)
```

[export/Export.py](export/Export.py) 中导出了一个 18 层的 ```ResNetConvFeatures``` 的模型 resnetc18-features.pt。
[deploy/src/Deploy.cpp](deploy/src/Deploy.cpp) 中的 ```testMultiOutput``` 函数加载了这个模型，使用假数据做了网络推理。
输出结果是 tuple 类型，通过遍历获取不同 stage 的卷积层特征。
```
    auto outputs = module->forward(inputs);
    printf("Is tuple: %d\n", outputs.isTuple());
    printf("Is tensor: %d\n", outputs.isTensor());
    printf("Is tensor list: %d\n", outputs.isTensorList());

    auto tuple = outputs.toTuple();
    auto elements = tuple->elements();
    for (auto& item : elements)
    {
        at::Tensor tensor = item.toTensor();
        std::cout << tensor.sizes() << "\n";
    }
```

# 多输入
[export/resnet.py](export/resnet.py) 中定义的 ```ResNetSiamese``` 是一个孪生网络，需要 2 个输入。
注意 ```ResNetSiamese``` 的 ```forward``` 函数的传入参数是 2 个，不能用 tuple 或者 list 组合起来。
```
    def forward(self, x0, x1):
        x0 = self.conv1(x0)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)
        ...
```

[export/Export.py](export/Export.py) 中导出了一个 18 层的 ```ResNetSiamese``` 的模型 resnets18-siamese.pt。
[deploy/src/Deploy.cpp](deploy/src/Deploy.cpp) 中的 ```testMultiInput``` 函数加载了这个模型，使用 2 组假数据组成的 vector 做了网络推理。
```
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({4, 3, 224, 224}));
    inputs.push_back(torch::ones({4, 3, 224, 224}));
```

# 其他

参见 [deploy/src/InferContext.h](deploy/src/InferContext.h)，里面给出了一个完整的加载单输入单输出模型做推理的例子，
还包括用 OpenCV 做数据预处理，支持 CPU 和 CUDA 做推理。
