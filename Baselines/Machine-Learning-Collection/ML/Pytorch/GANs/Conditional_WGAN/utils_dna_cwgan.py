import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
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

    def __init__(self, csv_file, root_dir, transform=None,train=False):
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
        if train:
            print ('length dataset:', int(len(self.strands)))
            self.strands = self.strands[:int(len(self.strands)-len(self.strands)/4)]
            print ('length train dataset:', int(len(self.strands)))
        else:
            print ('length dataset:', int(len(self.strands)))
            self.strands = self.strands[int(len(self.strands)-len(self.strands)/4):]
            print ('length test dataset:', int(len(self.strands)))

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

def Encoded_to_DNA(strands):
    mapping = {0:'A',1:'C',2:'G',3:'T'}

    strands_np = strands.permute(0,2,1)

    argmax = torch.argmax(strands_np, dim=2).tolist()

    mat = []
    for a in argmax:
        str = []
        for j in a:
            str.append(mapping[j])
        mat.append(''.join(str))

    return mat

def Test_environment_setup(name_main='1_test_model'):
    path = os.getcwd()
    path_conditional = path+'/DNA_project/Baselines/Machine-Learning-Collection/ML/Pytorch/GANs/Conditional_WGAN/'


    try:
        os.mkdir(path_conditional+name_main)
        os.mkdir(path_conditional+os.path.join(name_main,'generator'))
        os.mkdir(path_conditional+os.path.join(name_main,'extractor'))
        os.mkdir(path_conditional+os.path.join(name_main,'critic'))
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)

    path_generator = os.path.join(path_conditional,name_main,'generator')
    path_extractor = os.path.join(path_conditional,name_main,'extractor')
    path_critic = os.path.join(path_conditional,name_main,'critic')

    data_path = os.path.join(path,'DNA_project/Data/result/')
    data_file = os.path.join(path,'DNA_project/Data/result/clean_noisy_dataset_dev.txt')

    analytics_path = os.path.join(path_conditional,name_main,'alaytics.txt')

    return path,path_conditional,path_generator,path_extractor,path_critic,data_path,data_file,analytics_path
