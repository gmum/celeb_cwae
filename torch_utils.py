import torch

def to_gpu(x):
  if torch.cuda.is_available():
    x = x.cuda()
  return x
