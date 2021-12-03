import argparse
import os, glob
import imageio
import numpy as np
from tqdm import tqdm

import models

import torch
from torch import nn


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class PWD(nn.Module):
    '''
        Calculating pair-wise distances between two tensors.
    '''
    def __init__(self):
        super(PWD, self).__init__()

    def forward(self, p, pn, c, c_sq):
        p_rep = pn.repeat(1, len(c))     # (n / b) x k
        c_rep = c_sq.repeat(len(p), 1)     # (n / b) x k
        pc = torch.mm(p, c.t()).mul_(2)    # (n / b) x k

        dist = p_rep + c_rep - pc
        min_dists, min_labels = dist.min(1)

        return min_dists, min_labels

class Clustering():
    def __init__(
        self, k,
        n_init=1, max_iter=4500, symmetry='i', cpu=False, n_GPUs=1):
        '''
            input arguments
                n_init:
                    number of different trials for k-means clustering

                max_iter:
                    maximum iteration for k-means clustering

                symmetry:
                    transform invariant clustering parameter

                    i           default k-means clustering
                    ih          horizontal flip invariant clustering
                    ihvo        flip invariant clustering
                    ihvoIHVO    flip and rotation invariant clustering

                cpu and n_GPUs:
                    you can use multiple GPUs for k-means clustering
        '''

        self.k = k
        self.n_init = n_init
        self.max_iter = max_iter
        self.symmetry = symmetry
        self.device = torch.device('cpu' if cpu else 'cuda')
        if not cpu:
            self.n_GPUs = n_GPUs
            self.pairwise_dist = nn.DataParallel(PWD(), range(n_GPUs))
        self.tf_dict = {k: v for v, k in enumerate('ihvoIHVO')}

    def gen_feat(self, inp):
        with torch.no_grad():
            self.feat = self.encoder(inp)
        return self.feat

    def fit(self, points):
        points = points.to(self.device)
        n, n_feats = points.size()
        #self.make_sampler(n_feats, self.k)

        print('# points: {} / # parameters: {} / # clusters: {}'.format(
            n, points.nelement(), self.k)
        )

        print('Using {} initial seeds'.format(self.n_init))
        tqdm.monitor_interval = 0
        tqdm_init = tqdm(range(self.n_init), ncols=80)
        best = 1e8
        for i in tqdm_init:
            tqdm_init.set_description('Best cost: {:.4f}'.format(best))
            with torch.no_grad():
                centroids, labels, cost = self.cluster(points)
            if cost < best or i == 0:
                self.cluster_centers_ = centroids.clone()
                self.labels_ = labels.clone()
                best = cost
                best_idx = i

        print('Best round: {}'.format(best_idx))

        return self.cluster_centers_.cpu(), self.labels_.cpu()

    def cluster(self, points, log=False):
        n, n_feats = points.size()
        s = len(self.symmetry)

        labels = torch.LongTensor(n).to(self.device)
        ones = points.new_ones(n)
        init_seeds = ones.multinomial(self.k, replacement=False)

        pn = points.pow(2).sum(1, keepdim=True)
        centroids = points.index_select(0, init_seeds)
        tqdm_cl = tqdm(range(self.max_iter), ncols=80)

        # to prevent out of memory...
        if self.n_GPUs == 1:
            mem_check = self.k * n * n_feats
            mem_bound = 3 * 10**8
            if mem_check > mem_bound:
                split = round(mem_check / mem_bound)
            else:
                split = 1
        else:
            split = self.n_GPUs

        for _ in tqdm_cl:
            #centroids_full = self.transform(centroids.repeat(s, 1))
            centroids_full = centroids.repeat(split, 1)
            cn = centroids_full.pow(2).sum(1)

            if self.n_GPUs == 1:
                min_dists = []
                min_labels = []
                for _p, _pn, _c, _cn in zip(
                    points.chunk(split),
                    pn.chunk(split),
                    centroids_full.chunk(split),
                    cn.chunk(split)):

                    md, ml = self.pairwise_dist(_p, _pn, _c, _cn)
                    min_dists.append(md)
                    min_labels.append(ml)

                min_dists = torch.cat(min_dists)
                min_labels = torch.cat(min_labels)
            else:
                min_dists, min_labels = self.pairwise_dist(
                    points, pn, centroids_full, cn
                )

            cost = min_dists.mean().item()
            change = (min_labels != labels).sum().item()
            if change == 0: break

            tqdm_cl.set_description(
                'C: {:.3e} / Replace {}'.format(cost, change)
            )

            centroids_new = points.new_zeros(s * self.k, n_feats)
            centroids_new.index_add_(0, min_labels, points)
            #centroids_new = self.transform(centroids_new, inverse=True)
            centroids = sum(centroids_new.chunk(s))

            counts_new = points.new_zeros(s * self.k)
            counts_new.index_add_(0, min_labels, ones)
            counts = sum(counts_new.chunk(s))

            centroids.div_(counts.unsqueeze(-1))
            labels.copy_(min_labels)

        return centroids, labels, cost


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', default='dataset/DIV2K/DIV2K_valid_LR_bicubic')
    parser.add_argument('--hr_folder', default='dataset/DIV2K/DIV2K_valid_HR')
    parser.add_argument('--model', type=str, choices=['edsr', 'none'], help='super-resolution model')
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--n_buckets', type=int, default=8, help='Number of groups')
    parser.add_argument('--n_rounds', type=int, default=4, help='Number of clustering iterations')
    parser.add_argument('--window_size', type=int, default=32, help='Window size')
    parser.add_argument('--save_folder', default='output')
    args = parser.parse_args()
    
    
    color_code_R = {0: 0  , 1:  46, 2: 167, 3: 100, 4: 191, 5: 220, 6:   0, 7: 10}
    color_code_G = {0: 160, 1: 141, 2:   0, 3:  62, 4:  30, 5:  87, 6: 166, 7: 91}
    color_code_B = {0: 177, 1: 239, 2: 174, 3: 191, 4:  75, 5:  46, 6:   0, 7: 196}
    
    if not args.model == 'none':
        encoder_spec = {'name': 'edsr-baseline', 'args': {'no_upsampling': True, 'scale': args.scale}}
        encoder = models.make(encoder_spec).cuda()
    model = Clustering(args.n_buckets, n_init=args.n_rounds, n_GPUs=1)

    for path, hr_path in zip(sorted(glob.glob(os.path.join(args.input_folder, 'X{}'.format(args.scale), '*'))), sorted(glob.glob(os.path.join(args.hr_folder, '*')))):
        # read image
        print(path)
        image = imageio.imread(path)
        image = image[:64, :64]
        image_np = image
        image = torch.from_numpy(image).float().to('cuda')  # CHW-RGB to NCHW-RGB
        
        if not args.model == 'none':
            image = image.unsqueeze(0).permute(0,3,1,2)
            with torch.no_grad():
                image = encoder(image)
            image = image.squeeze(0).permute(1,2,0)
        
        H, W, _ = image.shape
        feature = image.reshape(H*W, -1)
        
        _, labels = model.fit(feature)
        group_feature = labels.reshape(H, W)
        group_R = np.vectorize(color_code_R.get)(group_feature).astype('uint8')
        group_G = np.vectorize(color_code_G.get)(group_feature).astype('uint8')
        group_B = np.vectorize(color_code_B.get)(group_feature).astype('uint8')
        
        hr_image = imageio.imread(hr_path)
        hr_image = hr_image[:64*args.scale, :64*args.scale]
        
        #Save image
        img_name = os.path.splitext(os.path.basename(path))[0]
        imgsave_folder = os.path.join(args.save_folder, img_name)
        if not os.path.exists(imgsave_folder):
            os.makedirs(imgsave_folder)
        
        imageio.imwrite(os.path.join(imgsave_folder, 'image_lr.png'), image_np)
        imageio.imwrite(os.path.join(imgsave_folder, 'image_hr.png'), hr_image)
        code_image = np.stack([group_R, group_G, group_B], axis=2)
        imageio.imwrite(os.path.join(imgsave_folder, 'kmeans.png'), code_image)


'''
if __name__ == '__main__':
    points_1 = torch.randn(100, 2)
    points_1[:, 0] = 2 * points_1[:, 0] + 1
    points_1[:, 1] = 4 * points_1[:, 1] + 1
    
    points_2 = torch.randn(100, 2)
    points_2[:, 0] = 10 * points_2[:, 0] + 5
    points_2[:, 1] = -2 * points_2[:, 1] - 5
    
    points_3 = torch.randn(100, 2)
    points_3[:, 0] = -3 * points_3[:, 0] -1
    points_3[:, 1] = 4 * points_3[:, 1] + 1
    
    points_4 = torch.randn(100, 2)
    points_4[:, 0] = -8 * points_4[:, 0] - 3
    points_4[:, 1] = -5 * points_4[:, 1] - 4
    
    points = torch.cat([points_1, points_2, points_3, points_4], dim=0)
    points = points.cuda()
    cl = Clustering(4, n_GPUs=1)
    
    clusters, labels = cl.fit(points)
    
    print(clusters)
    print(labels)
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    
    points_np = points.cpu().numpy()
    labels_np = labels.cpu().numpy()
    print(points.shape)
    print(labels.shape)
    
    plt.scatter(points_np[:, 0], points_np[:, 1], c=labels_np)
    plt.savefig('figure.png')
    plt.close()
'''