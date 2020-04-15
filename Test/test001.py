### 测试Pytorch进行内存连续的函数的作用

import torch

# 构造x为一维tensor
x = torch.arange(0, 6)
# 转置之后内存0 3 1 4 2 5不连续
y = x.view(2, 3).transpose(1, 0)
print(y)
# tensor([[0, 3],
#         [1, 4],
#         [2, 5]])
print(id(y.data) == id(x.data))
# True表示两者共享相同内存区
print(id(y.contiguous().data) == id(x.data))
# True表示两者共享相同内存区

# 无论访问第一个元素都是一样的首地址，tensor对外是一个整体，不能对其中一个元素进行地址获取
print(id(x[0]), id(x[5]), x[0], x[1])
print(id(y[0, 1]))

# 重排内存，方式为申请新的内存区，但是x和y依然共享一个内存区（相当于x和y同时换了一个）
y = y.contiguous()
print(id(x[0]), id(x[5]), x[0], x[1])
print(id(y[0, 1]))