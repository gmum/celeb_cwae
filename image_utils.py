import matplotlib.pyplot as plt
import random
import torch
import numpy as np


from dataset_utils import load_dataset

def plot_images(images, is_numpy=False, file_name=None):
    if not is_numpy:
        images = images.numpy()

    images_row = 6
    num_images = len(images)

    f, axarr = plt.subplots((num_images // images_row) + 1,
                            images_row, figsize=(10, 10))
    for ax in axarr.flatten():
        ax.axis('off')

    for i, image in enumerate(images):
        ax = axarr[i // images_row, i % images_row]
        # scale to silence matplitlib
        image = image - np.min(image)
        image = image / np.max(image)
        ax.imshow(image.transpose((1, 2, 0)))
    if file_name:
        plt.savefig('{}.png'.format(file_name))
    else:
        plt.show()

def crop_picture(picture, origin, width, height):
    # return pucture[channels, cropped width, cropped height]
    x = origin[0] * width
    y = origin[1] * height
    return picture[:, x:x+width, y:y+height]


def crop_batch(batch, origin, width, height):
    # assumes batch of size (num_batch * channel * x * y)
    batch = batch.numpy()
    # scale pictures
    results = [crop_picture(p, origin, width, height) for p in batch]
    # uncomment to upscale
    #results = np.kron(np.array(results), np.ones((2,2)))

    # scale positions
    pos_x = np.ones((batch.shape[2], batch.shape[3])
                    ) * np.arange(batch.shape[2])
    #pos_x = 2.*(pos_x - np.min(pos_x))/np.ptp(pos_x)-1
    pos_x = np.expand_dims(pos_x, axis=0)
    pos_x = crop_picture(pos_x, origin, width, height)
    pos_x = np.stack([pos_x] * batch.shape[0])

    pos_y = (np.ones((batch.shape[3], batch.shape[2]))
             * np.arange(batch.shape[3])).T
    #pos_y = 2.*(pos_y - np.min(pos_y))/np.ptp(pos_y)-1
    pos_y = np.expand_dims(pos_y, axis=0)
    pos_y = crop_picture(pos_y, origin, width, height)
    pos_y = np.stack([pos_y] * batch.shape[0])

    return (torch.FloatTensor(np.array(results)),
            torch.FloatTensor(pos_x),
            torch.FloatTensor(pos_y))


def get_crop_data(batch, width, height):
    # task specific
    crop = random.choice([[0, 0], [0, 1], [1, 0], [1, 1]])
    pictures, pos_x, pos_y = crop_batch(batch, crop, width, height)

    crop_channels_encoder = torch.cat([pos_x, pos_y], 1)

    crop_channels_decoder = crop_channels_encoder[:, :,
                                                  ::crop_channels_encoder.shape[2]-1,
                                                  ::crop_channels_encoder.shape[3]-1]

    # crop_channels_decoder = transform.resize(crop_channels_encoder.numpy(),
    #                                        [crop_channels_encoder.shape[0],
    #                                        crop_channels_encoder.shape[1],
    #                                        4,4])

    crop_channels_decoder = torch.FloatTensor(crop_channels_decoder)
    return pictures, crop_channels_encoder, crop_channels_decoder, crop


if __name__ == '__main__':

    data_loader = load_dataset('celeba', 32)

    sm_data, _ = next(iter(data_loader))
    sm_data, pos_encoder, pos_decoder, crop = get_crop_data(
        sm_data,  64, 64)
    plot_images(sm_data, file_name='crop_test')

    print(crop, sm_data.shape, pos_encoder.shape,
          pos_decoder.shape, pos_decoder)
