import torch

if not torch.cuda.is_available()):
    print("Cuda is not available")
else:
    print("Cuda is available")
