import torch
import torch.nn.init as init
import numpy as np

nonlinearity = ['linear', 'conv1d', 'conv2d', 'conv3d', 'sigmoid', 'tanh', 'relu', 'leaky_relu']

for func in nonlinearity:
    gain = init.calculate_gain(nonlinearity=func)
    # print(gain)
gain = init.calculate_gain(nonlinearity='leaky_relu', param=0.2)
# gain = init.calculate_gain('leaky_relu', 0.2)

# uniform
t1 = torch.Tensor(3, 1, 2)
init.uniform_(t1, a=2, b=3)

# normal
t2 = torch.Tensor(3, 1, 2)
init.normal_(t2, mean=0, std=1)

# constant
t3 = torch.tensor([3, 1], dtype=torch.int64)
init.constant_(t3, 2)
t4 = torch.Tensor(3, 1, 2) # dim
init.constant_(t4, 2)
t5 = torch.tensor(np.ndarray([3, 1], dtype=np.float), dtype=torch.int64)
init.constant_(t5, 2)

# eye - 2 dim
t6 = torch.Tensor(3, 3)
# t6 = torch.Tensor(3, 3, 3) # error
init.eye_(t6)

# ones/zeros
t7 = torch.empty(3, 1)
# print(t7.dtype)
init.ones_(t7)
print(t7)

# dirac



# xavier_normal

# kaiming_uniform

# kaiming_normal

# orthogonal

# sparse

