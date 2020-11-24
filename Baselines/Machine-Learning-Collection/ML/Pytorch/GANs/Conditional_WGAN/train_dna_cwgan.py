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
from sklearn.model_selection import train_test_split
sys.path.append('./DNA_project/Baselines/Machine-Learning-Collection/ML/Pytorch/GANs/Conditional_WGAN/')


from utils_dna_cwgan import gradient_penalty, save_checkpoint, load_checkpoint
import os
from model_dna_cwgan import Discriminator_1D, Generator_1D, initialize_weights,Dna_FeatureExtractor
from utils_dna_cwgan import ToTensor, DnaHotEncoding, StrandDataset,Encoded_to_DNA,Test_environment_setup
import numpy as np

#%%
# run file with pyton from home directory
# python ~/DNA_project/Baselines/Machine-Learning-Collection/ML/Pytorch/GANs/Conditional_WGAN/train_dna_cwgan.py

name_main = '1_test_model'
path,path_conditional,path_generator,path_extractor,path_critic,data_path,data_file,analytics_path = Test_environment_setup(name_main=name_main)
#%%
readme_name = 'readme.txt'
readme = open(path_conditional + os.path.join(name_main,readme_name), 'w')
readme.write('test x, dev dataset, looking to see if the algo is able to at least learn the primers')
readme.close()

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
strand_dataset_train = StrandDataset(csv_file=data_file,
                                    root_dir=data_path,transform = composed,train=True)
strand_dataset_test = StrandDataset(csv_file=data_file,
                                    root_dir=data_path,transform = composed,train=False)



loader = DataLoader(strand_dataset_train, batch_size=BATCH_SIZE, shuffle=True)

#%%


# initialize gen and disc, note: discriminator should be called critic,
# according to WGAN paper (since it no longer outputs between [0, 1])
gen = Generator_1D(Z_DIM, CHANNELS_IMG, FEATURES_GEN, test = False)
critic = Discriminator_1D(CHANNELS_IMG, FEATURES_CRITIC, test=False)
feature_extractor = Dna_FeatureExtractor(CHANNELS_IMG, FEATURES_CRITIC, test=False)

# gen.load_state_dict(torch.load('./DNA_project/Baselines/Machine-Learning-Collection/ML/Pytorch/GANs/Conditional_WGAN/models/generator/epoch1.pt'))
# critic.load_state_dict(torch.load('./DNA_project/Baselines/Machine-Learning-Collection/ML/Pytorch/GANs/Conditional_WGAN/models/critic/epoch1.pt'))
# feature_extractor.load_state_dict(torch.load('./DNA_project/Baselines/Machine-Learning-Collection/ML/Pytorch/GANs/Conditional_WGAN/models/extractor/epoch1.pt'))

gen = gen.to(device)
critic = critic.to(device)
feature_extractor = feature_extractor.to(device)

initialize_weights(gen)
initialize_weights(critic)
initialize_weights(feature_extractor)

#
# for n in loader:
#     noise = torch.randn(BATCH_SIZE,4, Z_DIM).to(device)
#     print ('noise shape',noise.shape)
#     device_n = n['clean'].to(device)
#     third_tensor = torch.cat((device_n, noise), 2)
#     print (third_tensor.shape)
#
#     embedding = feature_extractor(third_tensor)
#
#     fake = gen(embedding)
#     print ('fake size', fake.shape)
#     print (n['noisy'].shape)
#     a = n['noisy'].to(device)
#     x = critic(a)
#     print ('output D shape',x.shape)
#     sys.exit()



#%%

# initializate optimizer
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_extractor = optim.Adam(feature_extractor.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
metric_eucledian = nn.MSELoss()

# for tensorboard plotting
fixed_noise = torch.randn(32,4, Z_DIM).to(device)
fixed_strands = []
for n in loader:
    fixed_strands = n
    break

import shutil
shutil.rmtree(os.path.join(path,'logs/'))
writer_real = SummaryWriter(f"logs/GAN_MNIST/real")
writer_fake = SummaryWriter(f"logs/GAN_MNIST/fake")
step = 0

gen.train()
critic.train()
feature_extractor.train()
#%%
# analytics = open('./DNA_project/Baselines/Machine-Learning-Collection/ML/Pytorch/GANs/Conditional_WGAN/alaytics.txt','w')
# analytics_path = os.path.join(path_conditional,name_main,'alaytics.txt')
analytics = open(analytics_path,'w')
real_noisy_dna = Encoded_to_DNA(fixed_strands['noisy'][:32])
real_clean_dna = Encoded_to_DNA(fixed_strands['clean'][:32])
writer_real.add_text('CleanStrand:',real_clean_dna[0],str(1))
writer_real.add_text('NoisyStrand:',real_noisy_dna[0],str(1))

for epoch in range(NUM_EPOCHS):
    # Target labels not needed! <3 unsupervised
    print ('saving')
    # torch.save(gen.state_dict(), '~/DNA_project/Baselines/Machine-Learning-Collection/ML/Pytorch/GANs/Conditional_WGAN/models/generator/epoch'+str(epoch)+'.pt')
    # torch.save(feature_extractor.state_dict(), '~/DNA_project/Baselines/Machine-Learning-Collection/ML/Pytorch/GANs/Conditional_WGAN/models/extractor/epoch'+str(epoch)+'.pt')
    # torch.save(critic.state_dict(), '~/DNA_project/Baselines/Machine-Learning-Collection/ML/Pytorch/GANs/Conditional_WGAN/models/critic/epoch'+str(epoch)+'.pt')
    torch.save(gen.state_dict(), path_generator+'/epoch'+str(epoch)+'.pt')
    torch.save(feature_extractor.state_dict(), path_extractor+'/epoch'+str(epoch)+'.pt')
    torch.save(critic.state_dict(), path_critic+'/epoch'+str(epoch)+'.pt')

    print ('done saving')
    for batch_idx, real in enumerate(loader):



        noisy_real_device = real['noisy'].to(device)
        cur_batch_size = noisy_real_device.shape[0]
        # Train Critic: max E[critic(noisy_real_device)] - E[critic(fake)]
        # equivalent to minimizing the negative of that
        for _ in range(CRITIC_ITERATIONS):
            # noise = torch.randn(cur_batch_size, Z_DIM).to(device)

            # concat noise to real strand, pass noise in feature ex, get output and pass on gan for fake

            noise = torch.randn(cur_batch_size,4, Z_DIM).to(device) #adding different noise, but of normal distribution on the same batch sample? is this correct?
            clean_real_device = real['clean'].to(device)
            conditional_noise = torch.cat((clean_real_device, noise), 2)
            # parse the strand that is concatenated with normal noise through the feature extractor
            embedding = feature_extractor(conditional_noise)
            # pass embedding through the gan
            fake = gen(embedding)


            critic_real = critic(noisy_real_device).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            gp = gradient_penalty(critic, noisy_real_device, fake, device=device)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
            )
            mse_loss = metric_eucledian(noisy_real_device,fake)
            loss_final = loss_critic - mse_loss
            critic.zero_grad()
            feature_extractor.zero_grad()
            # update the feature extractor the same way as the critic? this is likely incorrect, it's just a guess for now xD
            loss_final.backward(retain_graph=True)
            opt_critic.step()

            # feature_extractor.zero_grad()
            # loss_critic.backward(retain_graph=True)
            # opt_extractor.step()
            # sys.exit()


        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        gen_fake = critic(fake).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Print losses occasionally and print to tensorboard
        if batch_idx % 20 == 0 and batch_idx > 0:
            analytics = open(analytics_path,'a')
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f} \n"
            )
            analytics.write(f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f} \n")

            with torch.no_grad():
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
                analytics.write(f"MSE:{mse}, SSIM: {sim_batch:.4f} \n")
                readme.close()
                # writer_real.add_image("Real", img_grid_real, global_step=step)
                # writer_fake.add_image("Fake", img_grid_fake, global_step=step)
                fake_dna = Encoded_to_DNA(fake[:32])

                fake_analysis = list(fake_dna[0])
                noisy_analysis = list(real_noisy_dna[0])
                # check how many nucleotides are the same
                equal_counter = 0
                for i in range(len(fake_analysis)):
                    if fake_analysis[i] == noisy_analysis[i]:
                        equal_counter +=1

                # do it over the batch
                total = 0
                for strand_id in range(len(fake_dna)):
                    strand = list(fake_dna[strand_id])
                    strand_counter = 0
                    for n_id in range(len(strand)):
                        if strand[n_id] == real_noisy_dna[strand_id][n_id]:
                            strand_counter +=1
                    total += strand_counter
                total = total/len(fake_dna)






                writer_fake.add_text('FakeStrand:',fake_dna[0] + ' | equal_counter:' + str(equal_counter)+'| batch_counter:'+str(total),str(epoch))



            step += 1


#%%
