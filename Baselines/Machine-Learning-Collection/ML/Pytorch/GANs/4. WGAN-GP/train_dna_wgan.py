"""
Training of WGAN-GP
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
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import pandas as pd
sys.path.append('./DNA_project/Baselines/Machine-Learning-Collection/ML/Pytorch/GANs/4. WGAN-GP/')
from utils import gradient_penalty, save_checkpoint, load_checkpoint

from model_dna import Discriminator_1D, Generator_1D, initialize_weights
from utils import ToTensor, DnaHotEncoding, StrandDataset
import numpy as np

# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 4
Z_DIM = 100
NUM_EPOCHS = 100
FEATURES_CRITIC = 100
FEATURES_GEN = 100
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10


# comment mnist above and uncomment below for training on CelebA
#dataset = datasets.ImageFolder(root="celeb_dataset", transform=transforms)


composed = transforms.Compose([DnaHotEncoding(),ToTensor()])
strand_dataset = StrandDataset(csv_file='./DNA_project/Data/result/clean_noisy_dataset_dev.txt',
                                    root_dir='./DNA_project/Data/result/',transform = composed)
loader = DataLoader(strand_dataset, batch_size=BATCH_SIZE, shuffle=True)


#%%


# initialize gen and disc, note: discriminator should be called critic,
# according to WGAN paper (since it no longer outputs between [0, 1])
gen = Generator_1D(Z_DIM, CHANNELS_IMG, FEATURES_GEN, test = False).to(device)
critic = Discriminator_1D(CHANNELS_IMG, FEATURES_CRITIC, test=False).to(device)
initialize_weights(gen)
initialize_weights(critic)
#
# for n in loader:
#     noise = torch.randn(BATCH_SIZE, Z_DIM).to(device)
#     print ('noise shape',noise.shape)
#     fake = gen(noise)
#     print (n['noisy'].shape)
#     a = n['noisy'].to(device)
#     x = critic(a)
#     print ('output D shape',x.shape)
#     sys.exit()

#%%

# initializate optimizer
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
metric_eucledian = nn.MSELoss()

# for tensorboard plotting
fixed_noise = torch.randn(32, Z_DIM).to(device)
writer_real = SummaryWriter(f"logs/GAN_MNIST/real")
writer_fake = SummaryWriter(f"logs/GAN_MNIST/fake")
step = 0

gen.train()
critic.train()
#%%
for epoch in range(NUM_EPOCHS):
    # Target labels not needed! <3 unsupervised
    for batch_idx, real in enumerate(loader):
        real_device = real['noisy'].to(device)
        cur_batch_size = real_device.shape[0]

        # Train Critic: max E[critic(real_device)] - E[critic(fake)]
        # equivalent to minimizing the negative of that
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn(cur_batch_size, Z_DIM).to(device)
            fake = gen(noise)


            critic_real = critic(real_device).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            gp = gradient_penalty(critic, real_device, fake, device=device)
            # print (gp,critic_real.shape,critic_fake.shape)
            # sys.exit()
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
            )
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()


        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        gen_fake = critic(fake).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Print losses occasionally and print to tensorboard
        if batch_idx % 100 == 0 and batch_idx > 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                # take out (up to) 32 examples
                # img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                # img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                print ('mse pytorch',metric_eucledian(real['noisy'][:32].to(device),fake[:32]))
                print ('mse skitlearn',mean_squared_error(real['noisy'][:32].detach().cpu().numpy(),fake[:32].detach().cpu().numpy()))
                end = []
                for i in range(32):
                    end.append(ssim(np.squeeze(np.moveaxis(real['noisy'][i].detach().cpu().numpy(), 0, 1)),np.squeeze(np.moveaxis(fake[i].detach().cpu().numpy(), 0, 1)),multichannel=True))
                sim_batch = sum(end)/len(end)
                print ('sim metric',sim_batch)
                
                # writer_real.add_image("Real", img_grid_real, global_step=step)
                # writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1
