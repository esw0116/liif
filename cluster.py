import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

import argparse
import glob, os
import imageio
import numpy as np
import tqdm
import yaml

import datasets
import models
import utils


def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, C, window_size, window_size)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(-1, C, window_size, window_size)

    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, C, window_size, window_size)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, -1, H // window_size, W // window_size, window_size, window_size)
    x = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, -1, H, W)
    return x


def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))


class Cluster(nn.Module):
    def __init__(self, encoder_spec, save_folder, hash_buckets=8, n_hashes=4, window_size=32):
        super().__init__()
        self.save_folder = save_folder
        self.hash_buckets = hash_buckets
        self.n_hashes = n_hashes
        self.window_size = window_size
        
        self.color_code_R = {0: 0  , 1:  46, 2: 167, 3: 100, 4: 191, 5: 220, 6:   0, 7: 10}
        self.color_code_G = {0: 160, 1: 141, 2:   0, 3:  62, 4:  30, 5:  87, 6: 166, 7: 91}
        self.color_code_B = {0: 177, 1: 239, 2: 174, 3: 191, 4:  75, 5:  46, 6:   0, 7: 196}

        self.encoder = models.make(encoder_spec).cuda()

    def gen_feat(self, inp):
        with torch.no_grad():
            self.feat = self.encoder(inp)
        return self.feat
    
    def LSH(self, hash_buckets, x):
        #x: [N,H*W,C]
        N = x.shape[0]
        device = x.device
        
        #generate random rotation matrix
        rotations_shape = (1, x.shape[-1], self.n_hashes, hash_buckets//2) #[1,C,n_hashes,hash_buckets//2]
        random_rotations = torch.randn(rotations_shape, dtype=x.dtype, device=device).expand(N, -1, -1, -1) #[N, C, n_hashes, hash_buckets//2]
        
        #locality sensitive hashing
        rotated_vecs = torch.einsum('btf,bfhi->bhti', x, random_rotations) #[N, n_hashes, H*W, hash_buckets//2]
        rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1) #[N, n_hashes, H*W, hash_buckets]
        
        #get hash codes
        hash_codes = torch.argmax(rotated_vecs, dim=-1) #[N,n_hashes,H*W]
        
        #add offsets to avoid hash codes overlapping between hash rounds 
        # offsets = torch.arange(self.n_hashes, device=device) 
        # offsets = torch.reshape(offsets * hash_buckets, (1, -1, 1))
        # hash_codes = torch.reshape(hash_codes + offsets, (N, -1,)) #[N,n_hashes*H*W]

        return hash_codes

    def forward(self, inp):
        # self.gen_feat(inp)
        # h,w = self.feat.shape[-2:]
        # feat_window = window_partition(self.feat, self.window_size)
        
        h,w = inp.shape[-2:]
        feat_window = window_partition(inp, self.window_size)
        
        N,_,H,W = feat_window.shape
        x_embed = feat_window.view(N,-1,H*W).contiguous().permute(0,2,1)
        
        #get assigned hash codes/bucket number         
        hash_window = self.LSH(self.hash_buckets, x_embed).detach() #[N,n_hashes, H*W]
        hash_window = hash_window.reshape(N, -1, H, W)
        hash_feature = window_reverse(hash_window, self.window_size, h, w).permute(0,2,3,1).cpu().numpy()

        #Color coding
        hash_image_R = np.vectorize(self.color_code_R.get)(hash_feature).astype('uint8')
        hash_image_G = np.vectorize(self.color_code_G.get)(hash_feature).astype('uint8')
        hash_image_B = np.vectorize(self.color_code_B.get)(hash_feature).astype('uint8')
        
        return hash_image_R, hash_image_G, hash_image_B


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', default='dataset/DIV2K/DIV2K_valid_LR_bicubic')
    parser.add_argument('--hr_folder', default='dataset/DIV2K/DIV2K_valid_HR')
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--save_folder')
    args = parser.parse_args()

    encoder_spec = {'name': 'edsr-baseline', 'args': {'scale': args.scale}}
    model = Cluster(encoder_spec=encoder_spec, save_folder=args.save_folder)

    for path, hr_path in zip(sorted(glob.glob(os.path.join(args.input_folder, 'X{}'.format(args.scale), '*'))), sorted(glob.glob(os.path.join(args.hr_folder, '*')))):
        # read image
        print(path)
        image = imageio.imread(path)
        image = image[:64, :64]
        image_np = image
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image).float().unsqueeze(0).to('cuda')  # CHW-RGB to NCHW-RGB
        
        hash_R, hash_G, hash_B = model(image)

        hr_image = imageio.imread(hr_path)
        hr_image = hr_image[:64*args.scale, :64*args.scale]
        
        #Save image
        img_name = os.path.splitext(os.path.basename(path))[0]
        imgsave_folder = os.path.join(args.save_folder, img_name)
        if not os.path.exists(imgsave_folder):
            os.makedirs(imgsave_folder)
        
        imageio.imwrite(os.path.join(imgsave_folder, 'image_lr.png'), image_np)
        imageio.imwrite(os.path.join(imgsave_folder, 'image_hr.png'), hr_image)
        n = hash_R.shape[-1]
        for i in range(n):
            code_image = np.stack([hash_R[0,:,:,i], hash_G[0,:,:,i], hash_B[0,:,:,i]], axis=2)
            imageio.imwrite(os.path.join(imgsave_folder, 'lsh_{:03d}.png'.format(i)), code_image)