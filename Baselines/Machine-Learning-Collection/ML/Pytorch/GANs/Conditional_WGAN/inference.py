
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
from sklearn.model_selection import train_test_split
sys.path.append('./DNA_project/Baselines/Machine-Learning-Collection/ML/Pytorch/GANs/Conditional_WGAN/')


from utils_dna_cwgan import gradient_penalty, save_checkpoint, load_checkpoint

from model_dna_cwgan import Discriminator_1D, Generator_1D, initialize_weights,Dna_FeatureExtractor
from utils_dna_cwgan import ToTensor, DnaHotEncoding, StrandDataset,Encoded_to_DNA
import numpy as np


#%%




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

# break up to train evaluation
composed = transforms.Compose([DnaHotEncoding(),ToTensor()])
strand_dataset_train = StrandDataset(csv_file='~/DNA_project/Data/result/clean_noisy_dataset_dev.txt',
                                    root_dir='~/DNA_project/Data/result/',transform = composed,train=True)
strand_dataset_test = StrandDataset(csv_file='~/DNA_project/Data/result/clean_noisy_dataset_dev.txt',
                                    root_dir='~/DNA_project/Data/result/',transform = composed,train=False)



loader = DataLoader(strand_dataset_train, batch_size=BATCH_SIZE, shuffle=True)

#%%


gen = Generator_1D(Z_DIM, CHANNELS_IMG, FEATURES_GEN, test = False)
critic = Discriminator_1D(CHANNELS_IMG, FEATURES_CRITIC, test=False)
feature_extractor = Dna_FeatureExtractor(CHANNELS_IMG, FEATURES_CRITIC, test=False)

gen.load_state_dict(torch.load('./DNA_project/Baselines/Machine-Learning-Collection/ML/Pytorch/GANs/Conditional_WGAN/1_train_models/generator/epoch1.pt'))
critic.load_state_dict(torch.load('./DNA_project/Baselines/Machine-Learning-Collection/ML/Pytorch/GANs/Conditional_WGAN/1_train_models/critic/epoch1.pt'))
feature_extractor.load_state_dict(torch.load('./DNA_project/Baselines/Machine-Learning-Collection/ML/Pytorch/GANs/Conditional_WGAN/models/extractor/epoch1.pt'))

gen.to(device)
critic.to(device)
feature_extractor.to(device)


#%%
metric_eucledian = nn.MSELoss()

# for tensorboard plotting
fixed_noise = torch.randn(32,4, Z_DIM).to(device)
fixed_strands = []
for n in loader:
    fixed_strands = n
    break


#%%
output = open('./DNA_project/Baselines/Machine-Learning-Collection/ML/Pytorch/GANs/Conditional_WGAN/qualitative_output.txt','w')
for batch_idx, real in enumerate(loader):
    fixed_conditional =  torch.cat((fixed_strands['clean'][:32].to(device), fixed_noise), 2)
    embedding = feature_extractor(fixed_conditional)
    fake = gen(embedding)
    # take out (up to) 32 examples
    # img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
    # img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
    # print ('mse pytorch',metric_eucledian(fixed_strands['noisy'][:32].to(device),fake[:32]))
    mse = metric_eucledian(fixed_strands['noisy'][:32].to(device),fake[:32]).item()
    # print ('mse skitlearn',mean_squared_error(fixed_strands['noisy'][:32].detach().cpu().numpy(),fake[:32].detach().cpu().numpy()))
    print (mse)
    end = []
    for i in range(32):
        end.append(ssim(np.squeeze(np.moveaxis(fixed_strands['noisy'][i].detach().cpu().numpy(), 0, 1)),np.squeeze(np.moveaxis(fake[i].detach().cpu().numpy(), 0, 1)),multichannel=True))
    sim_batch = sum(end)/len(end)
    print ('sim metric',sim_batch)
    fake_strands = Encoded_to_DNA(fake)
    real_strands = Encoded_to_DNA(fixed_strands['clean'][:32])
    noisy_strands = Encoded_to_DNA(fixed_strands['noisy'][:32])
    for i in range(32):
        output.write('id'+str(i)+'\n')
        output.write(noisy_strands[i]+'\n')
        output.write(fake_strands[i]+'\n')
    output.close()
    sys.exit()

    # analytics.write(f"MSE:{mse}, SSIM: {sim_batch:.4f} \n")
    # writer_real.add_image("Real", img_grid_real, global_step=step)
    # writer_fake.add_image("Fake", img_grid_fake, global_step=step)
