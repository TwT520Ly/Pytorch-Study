import torch

## 定义张量数据
t1 = torch.tensor([1, 2], dtype=torch.float16)
t2 = torch.tensor([1.5, 2]) # 默认torch.float32
t3 = torch.tensor([1, 2]) # 默认torch.int64
t1_ = torch.is_tensor(t1)
# print(t1) # tensor([1., 2.], dtype=torch.float16)
# print(t1[1]) # tensor(2., dtype=torch.float16)
# tensor可以直接访问下标并且可以直接进行强制类型转换
# print(float(t2[0])) # 1.5
# print(torch.is_floating_point(t1)) # True

## 修改默认类型(只能修改浮点数类型的默认值)
t4 = torch.tensor([1.2, 2])
# print(torch.get_default_dtype()) # torch.float32
# 仅仅对之后定义的浮点类型tensor有效
torch.set_default_dtype(torch.float64)
# print(torch.get_default_dtype()) # torch.float64
# print(t4.dtype) # torch.float32
t5 = torch.tensor([1.2, 3])
# print(t5.dtype) # torch.float64
torch.set_default_tensor_type(torch.FloatTensor)
t6 = torch.tensor([1.2, 2])
# print(t6.dtype) # torch.float32
tourch.set_default

##


