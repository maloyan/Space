import pickle
import json
import os
import sys

import segmentation_models_pytorch as smp
import ttach as tta
import albumentations as A
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from osgeo import gdal

from src.models.siamese_unet import SCSeResneXt, ResneXt
from src.dataset import Satellite
from src.utils import *

with open(sys.argv[1], 'r') as f:
    config = json.load(f)
df = pd.read_csv(config['sample_submission_path'])

best_model = torch.load(f"models/saved/{config['model_name']}.pth")
tta_model = tta.SegmentationTTAWrapper(
    best_model, 
    tta.aliases.d4_transform(), 
    merge_mode='mean'
)

original_res = []
res = []
for file in df['Id'].values:
    ds = gdal.Open(f"{config['images_path']}{file}.tif")

    IMG1 = np.array([ds.GetRasterBand(i).ReadAsArray() for i in range(1, 4)])
    IMG2 = np.array([ds.GetRasterBand(i).ReadAsArray() for i in range(5, 8)])
    IMG1, IMG2 = normalize(IMG1, IMG2, config['img_channels'])
    new_shape = generate_new_shape(IMG1, config['img_size'], config['img_channels'])
    res_mask = np.zeros((new_shape[0], new_shape[1]))

    if config['img_channels'] == 3:
        IMG1_new, IMG2_new = np.full(new_shape, IMG1[0][0][0]), np.full(new_shape, IMG1[0][0][0])
        IMG1_new[:IMG1.shape[0], :IMG1.shape[1], :] = IMG1
        IMG2_new[:IMG2.shape[0], :IMG2.shape[1], :] = IMG2
    else:
        IMG1_new, IMG2_new = np.full(new_shape, IMG1[0][0]), np.full(new_shape, IMG1[0][0])
        IMG1_new[:IMG1.shape[0], :IMG1.shape[1]] = IMG1
        IMG2_new[:IMG2.shape[0], :IMG2.shape[1]] = IMG2

    for i in range(0, new_shape[0], config['img_size']):
        for j in range(0, new_shape[1], config['img_size']):
            if config['is_siamese']:
                x_tensor = torch.Tensor(np.concatenate((
                    np.moveaxis(IMG1_new[i:i+config['img_size'], j:j+config['img_size'], :], -1, 0),
                    np.moveaxis(IMG2_new[i:i+config['img_size'], j:j+config['img_size'], :], -1, 0)
                ))).to(config['device']).unsqueeze(0)
            else:
                x_tensor = torch.Tensor(np.array([
                    IMG1_new[i:i+config['img_size'], j:j+config['img_size']] - \
                    IMG2_new[i:i+config['img_size'], j:j+config['img_size']]
                ])).to(config['device']).unsqueeze(0)
            pr_mask = tta_model(x_tensor)
            pr_mask = pr_mask.squeeze().detach().cpu().numpy()
            res_mask[i:i+config['img_size'], j:j+config['img_size']] = pr_mask

    res_mask = res_mask[:IMG1.shape[0], :IMG1.shape[1]]
    original_res.append(res_mask.astype(np.float16))
    res_mask =  res_mask > 0.4
    res.append(decode_mask(res_mask))

with open(f"predicted_masks/{config['model_name']}.pkl", 'wb') as f:
    pickle.dump(original_res, f)