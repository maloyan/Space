import numpy as np
import torch
from torch.utils.data import Dataset

class Satellite(Dataset):
    def __init__(
        self, 
        images,
        masks,
        augmentation=None,
        is_siamese=False,
        sampling_threshold=0.3
    ):
        self.images = images
        self.masks = masks
        self.augmentation = augmentation
        self.is_siamese = is_siamese
        self.sampling_threshold = sampling_threshold
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, i):
        img = self.images[i]
        mask = self.masks[i]
        
        flag = np.random.rand() > self.sampling_threshold
        
        if self.augmentation:
            sample = self.augmentation(image=img[0], image1=img[1], mask=mask)
            img1, img2, small_mask = sample['image'], sample['image1'], sample['mask']
            while small_mask.sum() == 0 and flag:
                sample = self.augmentation(image=img[0], image1=img[1], mask=mask)
                img1, img2, small_mask = sample['image'], sample['image1'], sample['mask']
        
        if self.is_siamese:
            return torch.tensor(np.concatenate((
                       np.moveaxis(img1, -1, 0),
                       np.moveaxis(img2, -1, 0))), dtype=torch.float), \
                   torch.tensor(np.array([small_mask]), dtype=torch.float)
        else:
            return torch.tensor(np.array([img1 - img2]), dtype=torch.float), \
                   torch.tensor(np.array([small_mask]), dtype=torch.float)