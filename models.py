from torch import nn
import torch
from skimage import transform

from torch_utils import to_gpu

class CelebEncoder(nn.Module):

    def __init__(self, input_channels, base_features, latent_size):
        super(CelebEncoder, self).__init__()

        self.input_channels = input_channels
        self.base_features = base_features
        self.latent_size = latent_size

        self.base = nn.Sequential(
            nn.Conv2d(input_channels, self.base_features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.base_features),
            nn.LeakyReLU(0.2),

            nn.Conv2d(self.base_features, self.base_features *
                      2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.base_features*2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(self.base_features*2, self.base_features *
                      4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.base_features*4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(self.base_features*4, self.base_features *
                      8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.base_features*8),
            nn.Tanh()
        )
        self.linear = nn.Linear(self.base_features*8 * 4 * 4, self.latent_size)

    def forward(self, x):
        x = self.base(x)
        x = x.view(-1, self.base_features*8 * 4 * 4)
        x = self.linear(x)
        return x


class CelebDecoder(nn.Module):

    def __init__(self, base_features, input_channels, output_channels):
        super(CelebDecoder, self).__init__()

        self.base_features = base_features
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        self.decoder_channels = None
        #self.linear = nn.Linear(LATENT_SIZE, self.base_features*8 * 8 * 8)

        self.deconv00 = nn.Sequential(
            nn.ConvTranspose2d(self.input_channels, self.base_features*8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(self.base_features*16),
            nn.ReLU()
        )

        self.deconv0 = nn.Sequential(
            nn.ConvTranspose2d(self.base_features*8,
                               self.base_features*8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(self.base_features*8),
            nn.ReLU()
        )

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(self.base_features*8,
                               self.base_features*4, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(self.base_features*4),
            nn.ReLU()
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(self.base_features*4,
                               base_features*2, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(self.base_features*2),
            nn.ReLU()
        )

        self.deconv3 = nn.Sequential(
            # state size. (ngf*4) x 7 x 7
            nn.ConvTranspose2d(self.base_features*2,
                               self.output_channels, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(self.base_features),
            nn.ReLU()
        )

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(self.base_features, self.output_channels, 1),
            nn.BatchNorm2d(self.output_channels),
            nn.Tanh(),
        )

    def set_decoder_channels(self, channels):
        self.decoder_channels = channels

    def scale_channels(self, channels, factor):
        channels_numpy = channels.numpy()
        scaled_channels = transform.resize(channels_numpy,
                                           [channels_numpy.shape[0],
                                            channels_numpy.shape[1],
                                            channels_numpy.shape[2] * factor,
                                            channels_numpy.shape[3] * factor,
                                            ])
        return torch.FloatTensor(scaled_channels)

    def forward(self, x):
        #x = self.linear(x)
        #x = x.view(x.shape[0],4, 4, 4)

        #position_channels = self.decoder_channels

        x = self.deconv00(x)
        x = self.deconv0(x)
        # add position data
        #position_channels = self.decoder_channels
        #x = torch.cat([x, to_gpu(position_channels)], 1)
        x = self.deconv1(x)

        # add position data
        #position_channels = self.scale_channels(self.decoder_channels, 2)
        #x = torch.cat([x, to_gpu(position_channels)], 1)
        x = self.deconv2(x)

        # add position data
        #position_channels = self.scale_channels(self.decoder_channels, 4)
        #x = torch.cat([x, to_gpu(position_channels)], 1)
        x = self.deconv3(x)

        #x = self.deconv4(x)
        return x


def create_models(dataset, encoder_config, decoder_config):
    if dataset == 'celeba':
        return to_gpu(CelebEncoder(**encoder_config)), to_gpu(CelebDecoder(**decoder_config))
    elif dataset == 'mnist':
        raise NotImplementedError()
    else:
        raise NotImplementedError()
