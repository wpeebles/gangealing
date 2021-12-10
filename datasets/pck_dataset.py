import torch
import os.path
from datasets.dataset import MultiResolutionDataset
from torchvision import transforms


_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)])


class PCKDataset(MultiResolutionDataset):
    # This dataset loads pairs of images and their key points. Its main purpose is for evaluating PCK-Transfer.
    # It can load either fixed, pre-determined pairs (useful for SPair-71K) or randomized pairs (useful for CUB).
    def __init__(self, path, transform=_transform, resolution=256, seed=0):
        super(PCKDataset, self).__init__(path, transform, resolution)

        keypoints_path = f'{path}/keypoints.pt'  # Required
        pairs_path = f'{path}/pairs.pt'  # Optional; if this file exists the dataset will sample fixed pairs
        permutation_path = f'{path}/permutation.pt'  # Optional

        assert os.path.isfile(keypoints_path), 'Could not find a keypoints.pt file'
        self.keypoints = torch.load(keypoints_path)

        if os.path.isfile(pairs_path):
            print('Found pairs.pt file; evaluating with fixed pairs')
            self.fixed_pairs = torch.load(pairs_path)
            self.pairs = self.fixed_pairs
            self.rng = None
        else:
            print('Could not find a pairs.pt file; evaluating with randomized pairs')
            assert seed is not None
            self.rng = torch.Generator()  # Should usually only be used when pairs.pt does not exist
            self.randomize_pairs(seed)

        if os.path.isfile(permutation_path):
            print('Found permutation.pt file')
            self.mirror_permutation = torch.load(permutation_path)
        else:
            print('Could not find permutation.pt file; will not permute left/right points on mirror')
            self.mirror_permutation = None

        # Some PCK datasets (e.g., SPair-71K) have per-image thresholds. Try to load them if they exist:
        thresholds_path = f'{path}/pck_thresholds.pt'  # Per-image thresholds (max bounding box size)
        inverse_ops_path = f'{path}/inverse_coordinates.pt'  # This contains information about per-image pre-processing
        assert os.path.isfile(thresholds_path) == os.path.isfile(inverse_ops_path)
        if os.path.isfile(thresholds_path):
            print('Using per-image PCK thresholds.')
            self.thresholds = torch.load(thresholds_path)
            self.inverse_ops = torch.load(inverse_ops_path)
        else:
            self.thresholds = None
            self.inverse_ops = None
        assert self.pairs.size(-1) == 2 and self.pairs.dim() == 2  # should be of size (N, 2)

    def randomize_pairs(self, seed=None):
        if self.rng is None:  # Pairs are fixed--should never randomize
            return
        if seed is not None:  # This seed should be the same across GPUs so the pairs are generated consistently
            self.rng.manual_seed(seed)
        indices = torch.randperm(super(PCKDataset, self).__len__(), generator=self.rng)
        if indices.size(0) % 2 == 1:  # Drop the last image if dataset size is odd so we can form pairs of two:
            indices = indices[:-1]
        self.pairs = indices.view(-1, 2)

    def randomize_fixed_pairs(self, seed=None):
        if seed is not None:
            rng = torch.Generator()
            rng.manual_seed(seed)
        else:
            rng = None
        indices = torch.randint(low=0, high=self.__len__(), size=(self.__len__(),), generator=rng)
        self.pairs = self.fixed_pairs[indices]
        assert self.pairs.size() == self.fixed_pairs.size()

    def __len__(self):
        return self.pairs.size(0)  # Number of pairs

    def __getitem__(self, index):
        ixA, ixB = self.pairs[index]
        ixA, ixB = ixA.item(), ixB.item()
        imgsA = super(PCKDataset, self).__getitem__(ixA)
        imgsB = super(PCKDataset, self).__getitem__(ixB)
        kpsA = self.keypoints[ixA]
        kpsB = self.keypoints[ixB]
        out = {'imgsA': imgsA, 'imgsB': imgsB, 'kpsA': kpsA, 'kpsB': kpsB, 'index': index}
        if self.thresholds is not None:  # Only used for SPair-71K categories:
            out['threshA'] = self.thresholds[ixA]
            out['scaleA'] = self.inverse_ops[ixA, 2]
            out['threshB'] = self.thresholds[ixB]
            out['scaleB'] = self.inverse_ops[ixB, 2]
        return out


def sample_infinite_pck_data(loader, seed=0):
    # Convenience function to make a data loader wrapped around PCKDataset "infinite."
    # After each full pass through the dataset (epoch), the pairs will be resampled (unless the pairs are fixed).
    rng = torch.Generator()
    rng.manual_seed(seed)  # This seed should be the same across GPUs for consistency
    BIG_NUMBER = 9999999999999
    while True:
        # Randomize pair indices before every epoch:
        pair_seed = torch.randint(0, BIG_NUMBER, (1,), generator=rng).item()
        loader.dataset.randomize_pairs(pair_seed)
        for batch in loader:
            yield batch
