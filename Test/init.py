import torch
import torch.nn.init as init
import numpy as np

# 第一种 torch
x = torch.ones(2, 2)
# 第二种 init
x = torch.empty(2, 2)
init.ones_(x)
# 第三种 tensor
x = torch.Tensor(2, 2)
x.fill_(1)
# 第四种 init
x = torch.Tensor(2, 2)
init.ones_(x)
# 第五种 torch
x = torch.tensor(np.ones(shape=(2, 2)).tolist())
