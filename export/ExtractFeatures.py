import torch
import resnet
import os

os.makedirs('../model', exist_ok=True)

featExtractor = resnet.resnetc18(pretrained=True)
device = torch.device('cpu')
featExtractor.to(device)
featExtractor.eval()

input = torch.zeros([4, 3, 224, 224], dtype=torch.float32)
output = featExtractor(input)
print(type(output))
for item in output:
    print(type(item), item.dim(), item.size())

tracedModule = torch.jit.trace(featExtractor, input)
tracedModule.save('../model/resnetc18-features.pt')

print()

siameseNetwork = resnet.resnets18(pretrained=True)
device = torch.device('cpu')
siameseNetwork.to(device)
siameseNetwork.eval()

input = [torch.zeros([4, 3, 224, 224], dtype=torch.float32),
         torch.zeros([4, 3, 224, 224], dtype=torch.float32)]
output = siameseNetwork(input[0], input[1])
print(type(output), output.dim(), output.size())

tracedModule = torch.jit.trace(siameseNetwork, input)
tracedModule.save('../model/resnets18-siamese.pt')