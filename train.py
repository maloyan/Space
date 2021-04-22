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

from src.models.siamese_unet import SCSeResneXt, ResneXt, DensenetUnet, DPNUnet
from src.models.snunet import SNUNet_ECAM
from src.dataset import Satellite
from src.utils import *

with open(sys.argv[1], 'r') as f:
    config = json.load(f)

if config['is_siamese']:
    if config['model'] == 'snunet':
        model = SNUNet_ECAM(out_ch=1)
    else:
        if config['model'] == 'seresnext50':
            model = SCSeResneXt(5, config['model'], reduction=2, mode='concat', num_channels=3, shared=True)
        elif config['model'] == 'resnext101':
            model = ResneXt(5, 'resnext101', shared=True)
        elif config['model'] == 'densenet161':
            model = DensenetUnet(5)
        elif config['model'] == 'dpn92':
            model = DPNUnet(5, shared=True)

        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(config['pretrained'])['state_dict'])
        model = model.module
        model.final = model.make_final_classifier(in_filters=config['in_filters'], num_classes=1)
else:
    model = smp.UnetPlusPlus(
        encoder_name=config["model"], 
        encoder_weights=config["pretrained"],
        in_channels=1,
        classes=1,
        activation=config["activation"]
    )

transform = A.Compose([
    A.RandomCrop(width=config['img_size'], height=config['img_size']),
    A.RandomRotate90(),
    A.Flip()
],
    additional_targets={'image0': 'image', 'image1': 'image'}
)

df = pd.read_csv(config['sample_submission_path'])

images = []
masks = []
for file in os.listdir(config['mask_path']):
    mask = gdal.Open(f"{config['mask_path']}{file}")
    mask = mask.GetRasterBand(1).ReadAsArray()

    ds = gdal.Open(f"{config['images_path']}{file}")
    img1, img2 = normalize(
        np.array([ds.GetRasterBand(i).ReadAsArray() for i in range(1, 4)]), 
        np.array([ds.GetRasterBand(i).ReadAsArray() for i in range(5, 8)]), 
        config['img_channels']
    )
    images.append([img1, img2])
    masks.append(mask)

train_dataset = Satellite(
    images * 100,
    masks * 100,
    augmentation=transform,
    is_siamese=config['is_siamese']
)

if config['is_siamese']:
    assert train_dataset[0][0].shape[0] == 2 * config['img_channels']
else:
    assert train_dataset[0][0].shape[0] == 1

train_loader = DataLoader(train_dataset, batch_size=config['batch'], shuffle=True, num_workers=12, drop_last=True)
valid_loader = DataLoader(train_dataset, batch_size=config['batch'], shuffle=True, num_workers=12, drop_last=True)

loss = smp.utils.base.SumOfLosses(
    smp.utils.losses.DiceLoss(),
    smp.utils.losses.BCELoss()
)

metrics = [
    smp.utils.metrics.IoU(threshold=0.5)
]

optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.0001),
])

train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=config['device'],
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=config['device'],
    verbose=True,
)

f = open(f"logs/{config['model_name']}", 'w+')

max_score = 0
patience = 0

for i in range(0, 150):    
    print('\nEpoch: {}'.format(i), file=f)
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)
    print(f"train_loss: {train_logs['dice_loss + bce_loss']}\t \
            val_loss:{valid_logs['dice_loss + bce_loss']}\t \
            IOU: {valid_logs['iou_score']}\n", file=f)


    optimizer.param_groups[0]['lr'] *= 0.97
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, f"models/saved/{config['model_name']}.pth")
        print('Model saved!', file=f)
        patience = 0
    else:
        patience += 1
    
    if patience == config['patience']:
        break

tta_model = tta.SegmentationTTAWrapper(
    torch.load(f"models/saved/{config['model_name']}.pth"), 
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