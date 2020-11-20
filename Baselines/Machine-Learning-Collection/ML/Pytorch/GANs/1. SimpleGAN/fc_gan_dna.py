import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
import pandas as pd
import os
import numpy as np
#%%
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        clean, noisy = sample['clean'], sample['noisy']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # image = image.transpose((2, 0, 1))
        return {'clean': torch.from_numpy(clean),
                'noisy': torch.from_numpy(noisy)}

class DnaHotEncoding(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        mapping = {'A':np.array((1,0,0,0)),'C': np.array((0, 1, 0, 0)),'G': np.array((0, 0, 1, 0)),'T':np.array((0, 0, 0, 1))  }
        clean, noisy = sample['clean'], sample['noisy']

        clean_hot = np.array([mapping[i] for i in list(clean)])
        noisy_hot = np.array([mapping[i] for i in list(noisy)])
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


class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, strand_dim,stride):
        super().__init__()
        # self.gen = nn.Sequential(
        #     nn.Linear(z_dim, 256),
        #     nn.LeakyReLU(0.01),
        #     nn.Linear(256, strand_dim),
        #     nn.Tanh(),  # normalize inputs to [-1, 1] so make outputs [-1, 1]
        # )
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.Conv2d(4, 20, 3, stride=2)
            nn.LeakyReLU(0.01),
            nn.Linear(256, strand_dim),
            nn.Tanh(),  # normalize inputs to [-1, 1] so make outputs [-1, 1]
        )

    def forward(self, x):
        return self.gen(x)


device = "cuda" if torch.cuda.is_available() else "cpu"
print (device)
lr = 3e-4
z_dim = 178
strand_dim = 178  # 784
batch_size = 32
num_epochs = 50
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
composed = transforms.Compose([DnaHotEncoding(),ToTensor()])
strand_dataset = StrandDataset(csv_file='./DNA_project/Data/result/clean_noisy_dataset_dev.txt',
                                    root_dir='./DNA_project/Data/result/',transform = composed)
loader = DataLoader(strand_dataset, batch_size=batch_size, shuffle=True)

print (strand_dataset[0]['noisy'].shape)

#%%
gen = Generator(z_dim, strand_dim,).to(device)
#%%
disc = Discriminator(strand_dim).to(device)
gen = Generator(z_dim, strand_dim).to(device)


opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()
writer_fake = SummaryWriter(f"logs/fake")
writer_real = SummaryWriter(f"logs/real")
step = 0

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise) # create 64 fake strands from gan
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        # where the second option of maximizing doesn't suffer from
        # saturating gradients
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step += 1

#%%
