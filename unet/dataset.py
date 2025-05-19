import glob
import os
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset



class FMRI3DDataset(Dataset):
    def __init__(self, root_dir, noise_std=0.1, preload_count=4):
        """
        Args:
            root_dir (str): Directory where the fMRI .nii.gz files are located.
            noise_std (float): Std. dev. of Gaussian noise added to the input.
            preload_count (int): Number of future slices to keep cached.
        """
        self.volume_paths = glob.glob(os.path.join(root_dir, '**/*.nii.gz'),
                                      recursive=True)
        self.samples = []  # list of (path, time_idx)
        self.noise_std = noise_std
        self.preload_count = preload_count

        # build sample list
        for path in self.volume_paths:
            nii = nib.load(path)
            data = nii.get_fdata()
            if data.ndim == 4:
                for t in range(data.shape[3]):
                    self.samples.append((path, t))

        # slice-level cache: keys are (path, t), values are 3D numpy arrays
        self.slice_cache = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, t = self.samples[idx]
        key = (path, t)

        if key not in self.slice_cache:
            self.slice_cache[key] = nib.load(path).get_fdata()[..., t]

        self._maintain_cache(idx)

        volume = self.slice_cache[key]

        # normalize
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

        noisy = np.zeros_like(volume)

        for z in range(volume.shape[2]):  # Iterate over slices in the 3rd dimension
            # Compute slice-specific noise level as a fraction of its max intensity
            max_intensity = np.max(volume[:, :, z])
            noise_std = np.random.uniform(0, 0.1) * max_intensity  # Scale noise

            # Add Gaussian noise to this slice
            noise = np.random.normal(0, noise_std, volume[:, :, z].shape)
            noisy[:, :, z] = volume[:, :, z] + noise
        

        x_noisy = torch.tensor(noisy, dtype=torch.float32).unsqueeze(0)
        x_clean = torch.tensor(volume, dtype=torch.float32).unsqueeze(0)

        return x_noisy, x_clean

    def _maintain_cache(self, current_idx):
        """
        Keep slices in the window [current_idx, current_idx + preload_count]
        loaded, and drop anything outside it.
        """
        # figure out which keys we want to keep
        desired = set()
        # include current
        desired.add(self.samples[current_idx])
        # include next preload_count
        for offset in range(1, self.preload_count + 1):
            next_idx = current_idx + offset
            if next_idx < len(self.samples):
                desired.add(self.samples[next_idx])

        # preload any missing ones
        for key in list(desired):
            if key not in self.slice_cache:
                path, t = key
                self.slice_cache[key] = nib.load(path).get_fdata()[..., t]

        # evict anything not in `desired`
        for key in list(self.slice_cache.keys()):
            if key not in desired:
                del self.slice_cache[key]
