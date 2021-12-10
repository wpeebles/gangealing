import torch
import torch.nn as nn
import numpy as np
import sklearn.decomposition
from utils.distributed import get_world_size, all_gather, rank0_to_all


class PCA:

    def __init__(self, n_components, w_batch):
        pca = sklearn.decomposition.IncrementalPCA(n_components)
        print('Fitting PCA...')
        pca.fit(w_batch.cpu().numpy())
        self.pca = pca

    def update(self, w_batch):
        self.pca.partial_fit(w_batch.cpu().numpy())

    def encode(self, x):
        encoded = self.pca.transform(x.cpu().numpy())
        encoded = torch.from_numpy(encoded).to(x.device)
        return encoded


class DirectionInterpolator(nn.Module):

    def __init__(self, pca_path, n_comps, inject_index, n_latent, num_heads=1, initializer=None):
        super().__init__()

        if pca_path is not None:
            with np.load(pca_path) as data:  # Static path from checkpoint
                lat_comp = torch.from_numpy(data['lat_comp'].squeeze(axis=1)).cuda()
                lat_mean = torch.from_numpy(data['lat_mean']).cuda()
                lat_std = torch.from_numpy(data['lat_stdev'][:, np.newaxis]).cuda()
                self.register_buffer('lat_mean', lat_mean)
                self.register_buffer('directions', lat_comp[:n_comps])
        else:  # These will be updated later; we randomly initialize them below so they can be loaded by a checkpoint if needed:
            self.register_buffer('directions', torch.randn(n_comps, 512))
            self.register_buffer('lat_mean', torch.randn(1, 512))

        with torch.no_grad():
            if initializer is None:
                initializer = torch.zeros(num_heads, n_comps, requires_grad=True)
            self.coefficients = nn.Parameter(initializer)

        self.n_latent = n_latent
        self.inject_index = inject_index
        self.num_heads = num_heads

    def forward(self, styled_latent, psi=None, lat_mean=None, pca=None, unfold=False):
        if pca is not None:
            self.assign_buffers(pca)
        else:
            return self.interpolate(styled_latent, psi, lat_mean, unfold)

    def interpolate(self, styled_latent, psi, lat_mean=None, unfold=False):
        assert len(styled_latent) == 1
        styled_latent = styled_latent[0]  # (N, 512)
        N = styled_latent.size(0)
        lat_mean = lat_mean if lat_mean is not None else self.lat_mean
        truncated_latents = lat_mean + (self.coefficients @ self.directions)  # (H, 512)
        truncated_latents = truncated_latents.repeat(N, 1)  # (N*H, 512)
        styled_latent = styled_latent.repeat_interleave(self.num_heads, dim=0)  # (N*H, 512)
        truncated_latents = truncated_latents.lerp(styled_latent, psi).unsqueeze(1).repeat(1, self.inject_index, 1)
        styled_latent = styled_latent.unsqueeze(1)
        fixed_latents = styled_latent.repeat(1, self.n_latent - self.inject_index, 1)
        out = torch.cat([truncated_latents, fixed_latents], dim=1)  # (N*H, n_latent, 512)
        if unfold:
            out = out.reshape(N, self.num_heads, self.n_latent, 512)
        return [out]

    @torch.no_grad()
    def assign_buffers(self, pca):
        lat_comp = torch.from_numpy(pca.pca.components_.astype(np.float32)).cuda()
        lat_mean = torch.from_numpy(pca.pca.mean_[np.newaxis].astype(np.float32)).cuda()
        self.register_buffer('directions', lat_comp)
        self.register_buffer('lat_mean', lat_mean)

    def assign_coefficients(self, initializer):
        initializer.requires_grad_(True)
        with torch.no_grad():
            self.coefficients.copy_(initializer)


@torch.no_grad()
def kmeans_plusplus(num_heads, num_latent, G, loss_fn, inject_index=6, batch_size=100):
    num_w_per_gpu = num_latent // get_world_size()
    batch_w = G.batch_latent(num_w_per_gpu)
    mean_w = all_gather(batch_w.mean(dim=0, keepdim=True)).mean(dim=0, keepdim=True)
    batch_fakes = []
    for i in range(0, num_w_per_gpu, batch_size):
        batch_w_in = batch_w[i:i+batch_size]
        fakes, _ = G([batch_w_in, mean_w.expand_as(batch_w_in)], input_is_latent=True, randomize_noise=True, inject_index=inject_index)
        batch_fakes.append(fakes.to('cpu'))
    batch_fakes = torch.cat(batch_fakes, 0)
    batch_w = all_gather(batch_w)  # Distribute these to every process (cheap compared to images)
    batch_w = batch_w.to('cpu')
    # Randomly pick the first centroid from the data:
    initial_w_idx = torch.randint(low=0, high=num_latent, size=(1,), device='cuda')
    initial_w_idx = rank0_to_all(initial_w_idx).item()  # Distribute the rank-0 sample
    dists = []
    centroid_idx = [initial_w_idx]
    for _ in range(num_heads - 1):
        # Recompute the current image of interest in each process:
        G_w, _ = G([batch_w[centroid_idx[-1]].unsqueeze(0).to('cuda'), mean_w], input_is_latent=True, randomize_noise=True, inject_index=inject_index)
        # Compute distance between the previous centroid and all other data points:
        dist = []
        for i in range(0, num_w_per_gpu, batch_size):
            dist_ = loss_fn(G_w.expand_as(batch_fakes[i:i+batch_size]), batch_fakes[i:i+batch_size].to('cuda')).squeeze()
            dist.append(dist_)
        dist = torch.cat(dist, 0)
        dist = all_gather(dist)
        dists.append(dist)
        # Compute the distance between all data points to their nearest centroid:
        closest = torch.stack(dists).min(dim=0).values
        logits_sqr = closest ** 2
        logits = logits_sqr / logits_sqr.sum()
        # Sample the next centroid, favoring data points that are poorly covered by the existing centroids:
        next_idx = rank0_to_all(torch.multinomial(logits, num_samples=1)).item()
        centroid_idx.append(next_idx)
    print(f'Centroids: {centroid_idx}')
    centroids = batch_w[centroid_idx].to('cuda')
    return centroids
