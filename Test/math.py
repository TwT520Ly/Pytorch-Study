import torch

# same dims && broadcast
x = torch.tensor([[1, 2], [1, 2]])
y = torch.tensor([[2, 1], [1, 2]])
print(x * y)
z = torch.mul(x, y)
print(z)

# only one element
x = torch.tensor([1, 2, 3])
y = 5
print(x * y)
z = torch.mul(x, y)
print(z)

# row or col (broadcast)
x = torch.tensor([1, 2, 3])
y = torch.tensor([2, 2, 2]).reshape(3, 1)
print(x * y)
z = torch.mul(x, y)
print(z)
x = torch.tensor([2, 2, 2]).reshape(3, 1)
y = torch.tensor([1, 2, 3])
print(x * y)
z = torch.mul(x, y)
print(z)

# matrix multiply - same dims
x = torch.tensor([[1, 2], [1, 2]])
y = torch.tensor([[2, 1], [1, 2]])
z = torch.mm(x, y)
print(z)
z = torch.matmul(x, y)
print(z)

# matrix multiply - different dims
x = torch.tensor([1, 2]).reshape(1, 2)
y = torch.tensor([[1, 1], [1, 2]])
# print(x.shape, y.shape)
z = torch.mm(x, y)
print(z)
z = torch.matmul(x, y)
print(z)

# https://pytorch.org/docs/stable/torch.html?highlight=torch%20mul#torch.matmul
x = torch.ones(3, 4)
y = torch.ones(5, 4, 2)
z = torch.matmul(x, y)
print(z.shape)