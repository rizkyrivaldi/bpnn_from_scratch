import torch

# Matrix Initialization
A = torch.cuda.DoubleTensor(
    [[1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]])

B = torch.cuda.DoubleTensor(
    [[3, 2, 1],
    [6, 5, 4],
    [9, 8, 7]]
)

C = torch.Tensor(
    [[1, 2, 3],
    [3, 2, 1],
    [4, 5, 6]]
).cuda()

D = torch.Tensor([[1, 2, 3], [3, 4, 5]]).cuda()

print(torch.mul(A, B))
print(A * B)
print(torch.matmul(A, B))
print(C.device)

B[:, 0] = torch.Tensor([1, 2, 3]).cuda()

print(B)
print(D**2)

print(torch.argmax(torch.Tensor([1, 6, 3])))