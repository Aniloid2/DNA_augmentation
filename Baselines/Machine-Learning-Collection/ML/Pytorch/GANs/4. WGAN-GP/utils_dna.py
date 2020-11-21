import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, C, H = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1)).repeat(1, C, H).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def save_checkpoint(state, filename="celeba_wgan_gp.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, gen, disc):
    print("=> Loading checkpoint")
    gen.load_state_dict(checkpoint['gen'])
    disc.load_state_dict(checkpoint['disc'])





class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        clean, noisy = sample['clean'], sample['noisy']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # image = image.transpose((2, 0, 1))
        return {'clean': torch.from_numpy(clean).float(),
                'noisy': torch.from_numpy(noisy).float()}

class DnaHotEncoding(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        mapping = {'A':np.array((1,0,0,0)),'C': np.array((0, 1, 0, 0)),'G': np.array((0, 0, 1, 0)),'T':np.array((0, 0, 0, 1))  }
        clean, noisy = sample['clean'], sample['noisy']

        # clean_hot = np.array([mapping[i] for i in list(clean)])
        # noisy_hot = np.array([mapping[i] for i in list(noisy)])
        clean_hot = np.moveaxis(np.array([mapping[i] for i in list(clean)]),0,1)
        noisy_hot = np.moveaxis(np.array([mapping[i] for i in list(noisy)]),0,1)
        # turn to noisy strands

        return {'clean': clean_hot,
                'noisy': noisy_hot}

class StrandDataset(Dataset):
    """Strand dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.strands =  pd.read_csv(csv_file, sep=",", header=0)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.strands)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()


        strand = list(self.strands.iloc[idx])
        clean = strand[-2].strip()
        noisy = strand[-1].strip()

        sample = {'clean': clean, 'noisy': noisy}

        if self.transform:
            sample = self.transform(sample)

        return sample
