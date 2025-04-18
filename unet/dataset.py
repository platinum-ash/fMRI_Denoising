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
            noise_std (float): Standard deviation of Gaussian noise added to the input data for denoising.
        """
        self.volume_paths = glob.glob(os.path.join(root_dir, '**/*.nii.gz'), recursive=True)
        self.samples = []
        self.noise_std = noise_std  # Standard deviation for Gaussian noise

        for path in self.volume_paths:
            nii = nib.load(path)
            data = nii.get_fdata()

            if data.ndim == 4:  # Only consider 4D fMRI volumes
                for t in range(data.shape[3]):
                    self.samples.append((path, t))

        self.volume_cache = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieves one 3D volume slice and adds noise to the input.
        """
        path, t = self.samples[idx]

        # Cache the volume to avoid repeated loading of large .nii files
        if path not in self.volume_cache:
            self.volume_cache[path] = nib.load(path).get_fdata()

        # Get the 3D volume slice at time step t
        volume = self.volume_cache[path][..., t]  # Shape: (X, Y, Z)

        # Normalize the volume to [0, 1]
        volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume) + 1e-8)

        # Add Gaussian noise slice-wise along Z (axis=2)
        # Get max per slice (shape: (1, 1, Z) for broadcasting)
        max_per_slice = np.max(volume, axis=(0, 1), keepdims=True)
        volume_noisy = np.zeros_like(volume)
        for z in range(volume.shape[2]):  # Iterate over slices in the 3rd dimension
            # Compute slice-specific noise level as a fraction of its max intensity
            max_intensity = np.max(volume[:, :, z])
            noise_std = np.random.uniform(0, self.noise_std) * max_intensity  # Scale noise

            # Add Gaussian noise to this slice
            noise = np.random.normal(0, noise_std, volume[:, :, z].shape)
            volume_noisy[:, :, z] = volume[:, :, z] + noise

        # Convert to torch tensors and add channel dimension
        x_noisy = torch.tensor(volume_noisy, dtype=torch.float32).unsqueeze(0)  # (1, X, Y, Z)
        x_clean = torch.tensor(volume, dtype=torch.float32).unsqueeze(0)        # (1, X, Y, Z)

        # Clear cache if it's too large (keep cache size under 10)
        if len(self.volume_cache) > 4:
            self.volume_cache.clear()

        return x_noisy, x_clean