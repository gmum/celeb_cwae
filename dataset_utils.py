
import os
import torch
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms, datasets
from skimage import io


class CelebaDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        # there are 202599 pics in dataset
        return 202598

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, '{:06d}.jpg'.format(idx+1))
        image = io.imread(img_name)
        # if image is greyscale, convert to rgb
        if len(image.shape) == 2:
            image = np.stack((image,)*3, axis=-1)
        if self.transform:
            sample = self.transform(image)
            # return 0 as fake label
        return sample, 0

def mnist_data():
    compose = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([.5], [.5])
        ])
    out_dir = './dataset'
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)

def celeba_data():
    compose = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([.5], [.5])
        ])
    data_dir = './celeba_resized'
    return CelebaDataset(root_dir=data_dir, transform=compose)


def load_dataset(dataset_name, batch_size):

    # Load data
    if dataset_name == 'celeba':
        data = celeba_data()
    elif dataset_name == 'mnist':
        data = mnist_data()
    else:
        raise NotImplementedError('select dataset')
  
    # Create loader with data, so that we can iterate over it
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    return data_loader
