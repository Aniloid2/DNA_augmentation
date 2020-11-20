"""
Training of DCGAN network on MNIST dataset with Discriminator
and Generator imported from models.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import sys
import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import numpy as np

sys.path.append('./DNA_project/Baselines/Machine-Learning-Collection/ML/Pytorch/GANs/2. DCGAN/')

from model import Discriminator, Generator, initialize_weights
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



# Hyperparameters etc.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-4  # could also use two lrs, one for gen and one for disc
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 1
NOISE_DIM = 100
NUM_EPOCHS = 5
FEATURES_DISC = 64
FEATURES_GEN = 64

#%%

transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
        ),
    ]
)

# If you train on MNIST, remember to set channels_img to 1
dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms,
                       download=True)

# comment mnist above and uncomment below if train on CelebA
#dataset = datasets.ImageFolder(root="celeb_dataset", transform=transforms)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
gen = Generator(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)
initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion = nn.BCELoss()
metric_eucledian = nn.MSELoss()

fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

gen.train()
disc.train()

#%%
for epoch in range(NUM_EPOCHS):
    # Target labels not needed! <3 unsupervised
    for batch_idx, (real, label) in enumerate(dataloader):



        real = real.to(device)
        noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)
        fake = gen(noise)


        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        D_x = loss_disc_real.mean().item()


        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)
        fake = gen(noise)
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()


        # get eucledian loss between the generated samples and the real samples as metric
        # get ssim between the generated sampes and the real samples

        # Print losses occasionally and print to tensorboard
        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} \
                  Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}, Loss D_x:{D_x:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(
                    real[:32], normalize=True
                )
                img_grid_fake = torchvision.utils.make_grid(
                    fake[:32], normalize=True
                )

                print ('mse pytorch',metric_eucledian(real[:32],fake[:32]))
                print ('mse skitlearn',mean_squared_error(real[:32].detach().cpu().numpy(),fake[:32].detach().cpu().numpy()))

                end = []
                for i in range(32):
                    end.append(ssim(np.squeeze(np.moveaxis(real[i].detach().cpu().numpy(), 0, 2)),np.squeeze(np.moveaxis(fake[i].detach().cpu().numpy(), 0, 2))))
                sim_batch = sum(end)/len(end)
                print ('batch sim skitlearn',sim_batch)

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1
