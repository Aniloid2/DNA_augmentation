"""
Discriminator and Generator implementation from DCGAN paper
"""

import torch
import torch.nn as nn


class Discriminator_1D(nn.Module):
    def __init__(self, channels_img, features_d,test = False):
        super(Discriminator_1D, self).__init__()
        # self.disc = nn.Sequential(
        #     # input: N x channels_img x 64 x 64
        #     nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
        #     nn.LeakyReLU(0.2),
        #     # _block(in_channels, out_channels, kernel_size, stride, padding)
        #     self._block(features_d, features_d * 2, 4, 2, 1),
        #     self._block(features_d * 2, features_d * 4, 4, 2, 1),
        #     self._block(features_d * 4, features_d * 8, 4, 2, 1),
        #     # After all _block img output is 4x4 (Conv2d below makes into 1x1)
        #     nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
        # )


        self.test = test
        self.layer1 = nn.Sequential(
            # input: N x channels_img x 64 x 64
            nn.Conv1d(
                channels_img, features_d, 1 , 1, padding=0
            ),
            nn.BatchNorm1d(features_d),
            nn.LeakyReLU(0.2),
            )
        self.layer2 = nn.Sequential(
        self._block(features_d, features_d, 5, 1, 0),
        self._block(features_d, features_d , 5, 1, 0),
        self._block(features_d, features_d , 5, 1, 0),
        self._block(features_d, features_d , 5, 1, 0),
        self._block(features_d, features_d , 5, 1, 0))

        self.layer3 = nn.Flatten()
        self.layer4 = nn.Linear(15700,1)


    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.InstanceNorm1d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        # return self.disc(x)
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
        # layer5 = self.layer5(layer4)
        # if self.test: print ('layer5 D:',layer5.shape)
        # layer6 = self.layer6(layer5)
        # if self.test: print ('layer6 D:',layer6.shape)
        return layer4


class Generator_1D(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g, test=False):
        super(Generator_1D, self).__init__()
        # self.net = nn.Sequential(
        #     # Input: N x channels_noise x 1 x 1
        #     self._block(channels_noise, features_g * 16, 4, 1, 0),  # img: 4x4
        #     self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
        #     self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
        #     self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
        #     nn.ConvTranspose2d(
        #         features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
        #     ),
        #     # Output: N x channels_img x 64 x 64
        #     nn.Tanh(),
        # )

        self.test = test

        self.layer1 = nn.Linear(channels_noise,channels_noise*177)
        self.layer2 = nn.Sequential(self._block(features_g, features_g,5, 1, 2),
        self._block(features_g, features_g,5, 1, 2),
        self._block(features_g, features_g,5, 1, 2 ),
        self._block(features_g, features_g,5, 1, 2 ),
        self._block(features_g, features_g,5, 1, 2 ))

        self.layer3 = nn.Sequential(
            # input: N x channels_img x 64 x 64
            nn.Conv1d(
                features_g, channels_img, 1 , 1, padding=0,padding_mode='replicate'
            ),
            nn.BatchNorm1d(channels_img),
            nn.LeakyReLU(0.2),
            )
        self.layer4 = nn.Softmax(dim=1)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,padding_mode='replicate',
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        # return self.net(x)
        if self.test: print ('input G:',x.shape)
        layer1 = self.layer1(x)
        layer1_prime = layer1.reshape(-1,100,177)
        if self.test: print ('layer1 G:',layer1.shape, layer1_prime.shape)
        layer2 = self.layer2(layer1_prime)
        if self.test: print ('layer2 G:',layer2.shape)
        layer3 = self.layer3(layer2)
        if self.test: print ('layer3 G', layer3.shape)
        layer4 = self.layer4(layer3)
        if self.test: print ('layer4 G',layer4.shape)
        return layer4


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def test():
    N, in_channels, H, W = 8, 3, 64, 64
    noise_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
    gen = Generator(noise_dim, in_channels, 8)
    z = torch.randn((N, noise_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"





# test()
