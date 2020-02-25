import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import collections

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(1, 20, 5)
        self.var = Variable(torch.randn([1]))
        self.par = nn.Parameter(torch.randn([1]))
        self.register_buffer('buffer', torch.randn([1]))

model = Model()
# buffer、parameter
print(model.state_dict().keys())

# print(torch.randn(1))
# print(torch.randn([1]))

model_1 = nn.Sequential(
    nn.Conv2d(1, 20, 5),
    nn.ReLU(),
    nn.Conv2d(20, 20, 5),
    nn.ReLU()
)

model_2 = nn.Sequential(
   collections.OrderedDict([
       ('conv1', nn.Conv2d(1, 20, 5)),
       ('relu1', nn.ReLU()),
       ('conv2', nn.Conv2d(20, 20, 5)),
       ('relu2', nn.ReLU())
   ])
)

for submodule in model_1.children():
    print(submodule)

for submodule in model_2.children():
    print(submodule)

class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])
        self.linears.extend(nn.ModuleList([nn.Linear(10, 10) for i in range(2)]))
        self.linears.append(nn.Linear(10, 10))

    def forward(self, x):
        for index, l in enumerate(self.linears):
            x = self.linears[index // 2](x) + l(x)
        return x


model_list = Model1()
print(model_list.state_dict().keys())