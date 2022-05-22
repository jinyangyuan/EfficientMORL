import torch
import numpy as np


class CustomizedDataset(torch.utils.data.Dataset):

    def __init__(self, images):
        super(CustomizedDataset, self).__init__()
        self.images = torch.from_numpy(np.rollaxis(images, -1, -3).astype(np.float32) / 255) * 2 - 1

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return {'imgs': self.images[idx]}
