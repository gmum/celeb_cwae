import torch
import numpy as np
from torch import nn, optim
from torch.distributions.normal import Normal
from tqdm import tqdm
import matplotlib.pyplot as plt
import math

from image_utils import get_crop_data, plot_images
from torch_utils import to_gpu



def plot_loss(losses, save=True):
    plt.plot(losses)
    plt.show()
    if save:
        plt.savefig('loss.png')
    

        

def train_loop(encoder, decoder, data_loader, config):

    data_config = config['data_config']
    image_w = data_config['image_width']
    image_h = data_config['image_height']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    latent_size = config['latent_size']
    num_batches = len(data_loader)
    
    D = image_w // 2 * image_h // 2
    gamma = (4/(3*batch_size)) ** (2/5)

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)
    loss_fn = nn.MSELoss()

    def cwae_forward(batch):

        cropped_pictures, crop_channels_en, crop_channels_de, _ = \
            get_crop_data(batch, image_w // 2, image_h // 2)

        cropped_batch = torch.cat([cropped_pictures, crop_channels_en], 1)
        cropped_batch = to_gpu(cropped_batch)

        # z = encoder(cropped_batch) # [BS x 64]
        decoder.set_decoder_channels(crop_channels_de)
        z = encoder(to_gpu(cropped_batch))
        z = torch.cat([z.view(z.shape[0], -1, 2, 2),
                       to_gpu(crop_channels_de)], 1)
        pred = decoder(z)
        return to_gpu(cropped_pictures), pred, z

    def train_cwae(batch):
        crop, pred, z = cwae_forward(batch)

        # MSE LOSS
        loss1 = loss_fn(crop, pred)
        # CWAE LOSS #1
        loss2 = 0

        flat_batch = crop.view(crop.shape[0], -1)  # [N, 64 * 64 * 3]
        distance = torch.unsqueeze(flat_batch, 0) - \
            torch.unsqueeze(flat_batch, 1)

        loss2 += torch.rsqrt(torch.add(torch.div(torch.norm(distance,
                                                            2, dim=2), 2*D-3), gamma)).sum()
        loss2 = torch.div(loss2, batch_size ** 2)

        # CWAE LOSS #2
        loss3 = 0
        loss3 += torch.rsqrt(torch.add(torch.div(torch.norm(flat_batch,
                                                            2, dim=1), 2*D - 3), gamma + 0.5)).sum()

        # for x in batch:
        #  loss3 += torch.rsqrt(torch.add(torch.div(torch.norm(x,2),2*D - 3), GAMMA + 0.5))
        loss3 = torch.mul(torch.div(loss3, batch_size), 2)
        loss3 = torch.add(loss3, (1 + gamma) ** (-0.5))

        loss = loss1 + \
            torch.log(torch.div(loss2 + loss3, 2 * math.sqrt(math.pi)))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    losses = []
    for epoch in range(num_epochs):
        global bch
        print("Epoch {}".format(epoch))
        with tqdm(total=num_batches) as bar:
            for batch, _ in data_loader:
                bch = batch
                loss = train_cwae(batch)
                losses.append(loss.cpu().detach().numpy())
                bar.update(1)

    plt.plot(losses)
    plt.show()

    print('recon')
    cropped_pictures, crop_channels_en, crop_channels_de, _ = \
        get_crop_data(bch, image_w // 2, image_h // 2)

    cropped_batch = torch.cat([cropped_pictures, crop_channels_en], 1)
    cropped_batch = to_gpu(cropped_batch)
    z = encoder(to_gpu(cropped_batch))
    decoder.set_decoder_channels(crop_channels_de)
    z = torch.cat([z.view(bch.shape[0], -1, 2, 2),
                   to_gpu(crop_channels_de)], 1)
    pred = decoder(z)
    plot_images(pred.detach().cpu())

    print('samples')
    sampler = Normal(0, 1)
    samples = sampler.rsample((batch_size, latent_size))
    fake_batch = torch.FloatTensor(np.ones((batch_size, 3,
                                            image_w, image_h)))
    _, _, crop_channels_de, crop = \
        get_crop_data(fake_batch, image_w // 2, image_h // 2)
    print(crop)
    decoder.set_decoder_channels(crop_channels_de)
    samples = torch.cat(
        [to_gpu(samples.view(batch_size, -1, 2, 2)), to_gpu(crop_channels_de)], 1)
    test_samples = decoder(to_gpu(samples))
    plot_images(test_samples.detach().cpu())
