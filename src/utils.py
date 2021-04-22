import numpy as np
import matplotlib.pyplot as plt

def decode_mask(mask):
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(30, 20))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

def normalize(img1, img2, num_channels):
    if num_channels == 1:
        img1 = img1[0] * 0.299 + img1[1] * 0.587 + img1[2] * 0.114
        img2 = img2[0] * 0.299 + img2[1] * 0.587 + img2[2] * 0.114

    img1 = img1 / 1024
    img2 = img2 / 1024
    
    img1 = (img1 - np.mean(img1)) / np.std(img1)
    img2 = (img2 - np.mean(img2)) / np.std(img2)
    
    if num_channels == 3:
        img1 = np.moveaxis(img1, 0, -1)
        img2 = np.moveaxis(img2, 0, -1)

    return img1, img2

def generate_new_shape(img, img_size, num_channels):
    if num_channels == 1:
        new_shape = (
            int(np.ceil(img.shape[0] / img_size) * img_size), 
            int(np.ceil(img.shape[1] / img_size) * img_size)
        )
    else:
        new_shape = (
            int(np.ceil(img.shape[0] / img_size) * img_size), 
            int(np.ceil(img.shape[1] / img_size) * img_size),
            3
        )
    return new_shape