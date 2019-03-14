import torch
import resnet

featExtractor = resnet.resnetc50(pretrained=True)
device = torch.device('cpu')
featExtractor.to(device)
featExtractor.eval()

input = torch.zeros([4, 3, 224, 224], dtype=torch.float32)
output = featExtractor(input)
print(type(output))
for item in output:
    print(type(item), item.dim(), item.size())

tracedModule = torch.jit.trace(featExtractor, input)
tracedModule.save('resnet50-features.pt')