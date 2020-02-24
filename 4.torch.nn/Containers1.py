import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))

class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.add_module("conv1", nn.Conv2d(1, 20, 5))

class Model3(Model1):
    def __init__(self):
        super(Model3, self).__init__()
        # base on model1 and add the "conv3"
        sub_module = nn.Conv2d(20, 20, 5)
        self.add_module("conv3", sub_module)
        self.add_module("conv4", sub_module)

model2 = Model2()
print(model2.conv1)

model3 = Model3()
print(model3.conv2)
print(model3.conv3)

# print(model3.children())
print('-----')
for sub_module in model3.children():
    print(sub_module)

print('-----')
for sub_module in model3.modules():
    print(sub_module)

print('-----')
for name, module in model3.named_children():
    if name in ['conv3']:
        print(True)

print('-----')
for name, param in model3.named_parameters():
    if name in ['conv3.weight', 'conv3.bias']:
        print(type(param.data), param.size())

# model3.cpu()
# model3.cuda(device=1)

# model3.double()
# model3.float()
# model3.half()

## Dropout、BN
# model3.eval()
# model3.train()
