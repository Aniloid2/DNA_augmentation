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
import pandas as pd
sys.path.append('./DNA_project/Baselines/Machine-Learning-Collection/ML/Pytorch/GANs/2. DCGAN/')

# from model_dna import Discriminator_1D, Generator_1D, initialize_weights

class Discriminator_1D(nn.Module):
    def __init__(self, channels_img , features_d,kernel_size=4,stride=2, test=False):
        super(Discriminator_1D, self).__init__()
        self.test = test
        self.layer1 = nn.Sequential(
            # input: N x channels_img x 64 x 64
            nn.Conv1d(
                4, 100, 1 , 1, padding=0
            ),
            nn.BatchNorm1d(100),
            nn.LeakyReLU(0.2),
            )
        self.layer2 = nn.Sequential( self._block(100, 100, 5, 1, 0), self._block(100, 100 , 5, 1, 0),self._block(100, 100 , 5, 1, 0),self._block(100, 100 , 5, 1, 0),self._block(100, 100 , 5, 1, 0))
        # self.layer3 = self._block(100, 100 , 5, 1, 0)
        # self.layer4 = self._block(100, 100, 5, 1, 0)
        # self.layer5 = self._block(100, 100 , 5, 1, 0)
        self.layer3 = nn.Flatten()
        self.layer4 = nn.Linear( 15700,1)
        self.layer5 = nn.Sigmoid()



        # self.layer1 = nn.Sequential(
        #     # input: N x channels_img x 64 x 64
        #     nn.Conv1d(
        #         channels_img, features_d, kernel_size , stride , padding=0
        #     ),
        #     nn.LeakyReLU(0.2),
        #     )
        # self.layer2 = self._block(features_d, features_d, kernel_size, stride, 0)
        # self.layer3 = self._block(features_d, features_d , kernel_size, stride, 0)
        # self.layer4 = self._block(features_d, features_d , kernel_size, stride, 0)
        # self.layer5 = nn.Conv1d(features_d, 1, 8, stride, padding=0)
        # self.layer6 = nn.Sigmoid()



        # self.disc = nn.Sequential(
        #     # input: N x channels_img x 64 x 64
        #     nn.Conv1d(
        #         channels_img, features_d, kernel_size , stride , padding=0
        #     ),
        #     nn.LeakyReLU(0.2),
        #     # _block(in_channels, out_channels, kernel_size, stride, padding)
        #     self._block(features_d, features_d, kernel_size, stride, 0),
        #     self._block(features_d, features_d , kernel_size, stride, 0),
        #     self._block(features_d , features_d , kernel_size, stride, 0),
        #
        #     # After all _block img output is 4x4 (Conv2d below makes into 1x1)
        #     nn.Conv1d(features_d, 1, 8, stride, padding=0),
        #     nn.Sigmoid(),
        # )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        # return self.disc(x)
        if self.test: print (x.shape)
        layer1 = self.layer1(x)
        if self.test: print ('layer1 D:',layer1.shape)
        layer2 = self.layer2(layer1)
        if self.test: print ('layer2 D:',layer2.shape)
        layer3 = self.layer3(layer2)
        if self.test: print ('layer3 D:',layer3.shape)
        layer4 = self.layer4(layer3)
        if self.test: print ('layer4 D:',layer4.shape)
        layer5 = self.layer5(layer4)
        if self.test: print ('layer5 D:',layer5.shape)
        # layer6 = self.layer6(layer5)
        # if self.test: print ('layer6 D:',layer6.shape)
        return layer5


class Generator_1D(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g, test=False):
        super(Generator_1D, self).__init__()
        self.test = test

        self.layer1 = nn.Linear(100,100*197)
        self.layer2 = nn.Sequential(self._block(100, 100,5, 1, 0),self._block(100, 100,5, 1, 0),self._block(100, 100,5, 1, 0),self._block(100, 100,5, 1, 0),self._block(100, 100,5, 1, 0))
        # self.layer2 = self._block(100, 100,5, 1, 0)
        # self.layer3 = self._block(100, 100,5, 1, 0)
        # self.layer4 = self._block(100, 100,5, 1, 0)
        self.layer3 = nn.Sequential(
            # input: N x channels_img x 64 x 64
            nn.Conv1d(
                100, 4, 1 , 1, padding=0,padding_mode='replicate'
            ),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.2),
            )
        self.layer4 = nn.Softmax(dim=1)





        # self.layer1 = self._block(177, 87,9, 2, 0) # img: 4x4
        # self.layer2 = self._block(87, 42, 4, 2, 0)  # img: 8x8
        # self.layer3 = self._block(42, 20, 4, 2, 0)  # img: 8x8
        # self.layer4 = self._block(20, 9, 5, 2, 0)  # img: 8x8
        # self.layer5 = self._block(9, 4, 5, 2, 0)  # img: 8x8
        # self.layer6 = nn.Tanh()
        # self.net = nn.Sequential(
        #     # Input: N x channels_noise x 1 x 1
        #     self._block(channels_noise, features_g * 16, 4, 1, 0),  # img: 4x4
        #     self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
        #     self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
        #     self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
        #     nn.ConvTranspose1d(
        #         features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
        #     ),
        #     # Output: N x channels_img x 64 x 64
        #     nn.Tanh(),
        # )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            # nn.ConvTranspose1d(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                padding_mode = 'replicate',
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        if self.test: print ('input G:',x.shape)
        layer1 = self.layer1(x)
        layer1_prime = layer1.reshape(-1,100,197)
        if self.test: print ('layer1 G:',layer1.shape, layer1_prime.shape)
        layer2 = self.layer2(layer1_prime)
        if self.test: print ('layer2 G:',layer2.shape)
        layer3 = self.layer3(layer2)
        if self.test: print ('layer3 G', layer3.shape)
        layer4 = self.layer4(layer3)
        if self.test: print ('layer4 G',layer4.shape)
        # layer5 = self.layer5(layer4)
        # if self.test: print ('layer5 G',layer5.shape)
        # layer6 = self.layer6(layer5)
        # if self.test: print ('layer6 G',layer6.shape)
        return layer4
        # return self.net(x)






def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.BatchNorm1d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


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
        mapping = {'A':np.array((1,0 ,0,0)) ,'C': np.array((0 , 1 , 0 , 0 )) ,'G': np.array((0 , 0 , 1 , 0 )) ,'T':np.array((0 , 0, 0, 1))  }
        clean, noisy = sample['clean'], sample['noisy']



        clean_hot = np.moveaxis(np.array([mapping[i] for i in list(clean)]),0,1)

        noisy_hot = np.moveaxis(np.array([mapping[i] for i in list(noisy)]),0,1)
        # clean_hot =  np.array([mapping[i] for i in list(clean)])
        # noisy_hot =  np.array([mapping[i] for i in list(noisy)])
        # turn to noisy strands

        return {'clean': clean_hot,
                'noisy': noisy_hot}


# mapping = {'A':np.array((1,0 ,0,0)).astype(float),'C': np.array((0 , 1 , 0 , 0 )).astype(float),'G': np.array((0 , 0 , 1 , 0 )).astype(float),'T':np.array((0 , 0, 0, 1)).astype(float)  }
# print (mapping)



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
BATCH_SIZE = 32
IMAGE_SIZE = 177
CHANNELS_IMG = 4
NOISE_DIM = 100
NUM_EPOCHS = 5
FEATURES_DISC = 177
FEATURES_GEN = 64


composed = transforms.Compose([DnaHotEncoding(),ToTensor()])
# strand_dataset = StrandDataset(csv_file='./DNA_project/Data/result/clean_noisy_dataset_dev.txt',
#                                     root_dir='./DNA_project/Data/result/',transform = composed)

strand_dataset = StrandDataset(csv_file='./DNA_project/Data/result/clean_noisy_dataset_dev.txt',
                                    root_dir='./DNA_project/Data/result/',transform = composed)
loader = DataLoader(strand_dataset, batch_size=BATCH_SIZE, shuffle=True)


#%%

gen = Generator_1D(177, 4, 64, test=False).to(device)
# gen = Generator_1D(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
# disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)

disc = Discriminator_1D(4, 4, test=False).to(device)
# print (disc)
initialize_weights(gen)
initialize_weights(disc)
#
for n in loader:
    noise = torch.randn(BATCH_SIZE, NOISE_DIM).to(device)
    print ('noise shape',noise.shape)
    fake = gen(noise)
    print (n['noisy'].shape)
    a = n['noisy'].to(device)
    x = disc(a)
    print ('output D shape',x.shape)
    sys.exit()

#%%

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion = nn.BCELoss()
# criterion = nn.CrossEntropyLoss()
metric_eucledian = nn.MSELoss()

fixed_noise = torch.randn(32, NOISE_DIM).to(device)
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

gen.train()
disc.train()

#%%
for epoch in range(NUM_EPOCHS):
    # Target labels not needed! <3 unsupervised
    for batch_idx,  real in enumerate(loader):


        real_device = real['noisy'].to(device)

        noise = torch.randn(BATCH_SIZE, NOISE_DIM).to(device)
        fake = gen(noise)




        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))

        disc_real = disc(real_device).reshape(-1)

        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).reshape(-1)

        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) /5
        disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        D_x = loss_disc_real.mean().item()


        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        noise = torch.randn(BATCH_SIZE, NOISE_DIM).to(device)
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
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}, Loss D_x:{D_x:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                # take out (up to) 32 examples
                # img_grid_real = torchvision.utils.make_grid(
                #     real[:32], normalize=True
                # )

                # img_grid_fake = torchvision.utils.make_grid(
                #     fake[:32], normalize=True
                # )

                print ('mse pytorch',metric_eucledian(real['noisy'][:32].to(device),fake[:32]))
                # print ('real',real['noisy'][0])

                # print ('fake',fake[0])

                print ('mse skitlearn',mean_squared_error(real['noisy'][:32].detach().cpu().numpy(),fake[:32].detach().cpu().numpy()))

                # print (np.moveaxis(real[i].detach().cpu().numpy(),0,1).shape)
                # sys.exit()

                end = []
                for i in range(32):
                    end.append(ssim(np.squeeze(np.moveaxis(real['noisy'][i].detach().cpu().numpy(), 0, 1)),np.squeeze(np.moveaxis(fake[i].detach().cpu().numpy(), 0, 1)),multichannel=True))
                sim_batch = sum(end)/len(end)
                print ('sim metric',sim_batch)
                # sys.exit()
                # writer_real.add_image("Real", img_grid_real, global_step=step)
                # writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1
#%%
