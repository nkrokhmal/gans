import torch.nn.functional as F
import torch

if __name__ == '__main__':
    x = torch.randn(1, 1, 4, 4)
    print(x)
    print()
    result = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
    print(result.shape)
    print(result)