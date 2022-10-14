import os
import sys
sys.path.insert(0, "..")
import torch
from utils.splat2d_cpu import splat2d
# import pytest


def test_splat_cpu():
    
    rs = 64   # resolution
    blank_img = torch.zeros([1, 3, 512, 512],device='cpu')
    points = torch.rand([1, 943, 2],dtype = torch.float32,device='cpu')
    colors = torch.rand([1, 943, 3],dtype = torch.float32,device='cpu')
    sigma = torch.tensor([0.3000], device='cpu')
    prop_obj_img = splat2d(blank_img, points, colors, sigma, False)  # (N, C, H, W)
    # import pdb;pdb.set_trace()

    assert prop_obj_img!=blank_img

    # assert im.shape == (1,3,512,512)
    # #torch.save({"image":im, "z":z}, "fixtures/stylegan_ref.pth")
    # #import pdb; pdb.set_trace()
    # assert torch.allclose(im, ref["image"], atol=1e-4)


if __name__ == '__main__':
    test_splat_cpu()
