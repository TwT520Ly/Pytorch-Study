import torch
import torch.nn as nn
from torch.autograd import Variable

conv1 = nn.Conv2d(9, 12, 3, 1, padding=1, groups=3)
input = Variable(torch.randn([1, 9, 5, 5]))
output = conv1(input)
print(output.size())

cnt = 0
for param in conv1.parameters():
    cnt += 1
    print(param.size())
print(cnt)