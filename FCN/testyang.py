import torch


model = torch.load("xxx.pth")
print(type(model))

model.forward()