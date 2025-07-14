import glob
import os
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset


class FMRI3DDataset(Dataset):
    def __init__(self, root_dir, noise_std=0.1):
        """
        Args:
            root_dir (str): Directory where the fMRI .nii.gz files are located.
            noise_std (float): Max std. dev. of Gaussian noise added to the input.
        """
        self.volume_paths = glob.glob(os.path.join(root_dir, '**/*.nii.gz'), recursive=True)
        self.noise_std = noise_std
        self.samples = []  # list of (path, time_idx)

        for path in self.volume_paths:
            nii = nib.load(path)
            data = nii.get_fdata()
            if data.ndim == 4:
                for t in range(data.shape[3]):
                    self.samples.append((path, t))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, t = self.samples[idx]

        # Load the 3D volume at time index t
        volume = nib.load(path).get_fdata()[..., t]

        # Normalize to [0, 1]
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

        # Add slice-wise noise
        noisy = np.zeros_like(volume)
        for z in range(volume.shape[2]):
            max_intensity = np.max(volume[:, :, z])
            noise_std = np.random.uniform(0, self.noise_std) * max_intensity
            noise = np.random.normal(0, noise_std, volume[:, :, z].shape)
            noisy[:, :, z] = volume[:, :, z] + noise

        # Clip to [0, 1] after noise
        noisy = np.clip(noisy, 0.0, 1.0)

        # Convert to torch tensors with shape [1, D, H, W]
        x_noisy = torch.tensor(noisy, dtype=torch.float32).unsqueeze(0)
        x_clean = torch.tensor(volume, dtype=torch.float32).unsqueeze(0)

        return x_noisy, x_clean
