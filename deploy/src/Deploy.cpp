#include <stdio.h>

#include <iostream>
#include <memory>
#include <string>

#include "torch/script.h"

int testMultiOutput()
{
    std::string modelPath = "/home/zhengxuping/Projects/PyTorchDeploy/model/resnetc18-features.pt";
    std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(modelPath);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({8, 3, 224, 224}));

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

    return 0;
}

int testMultiInput()
{
    std::string modelPath = "/home/zhengxuping/Projects/PyTorchDeploy/model/resnets18-siamese.pt";
    std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(modelPath);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({4, 3, 224, 224}));
    inputs.push_back(torch::ones({4, 3, 224, 224}));

    auto outputs = module->forward(inputs);
    at::Tensor tensor = outputs.toTensor();
    std::cout << tensor.sizes() << "\n";

    return 0;
}

int main()
{
    printf("Test multi input\n");
    testMultiInput();
    printf("\n");
    printf("Test multi output\n");
    testMultiOutput();
    return 0;
}