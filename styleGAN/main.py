from __future__ import print_function
import argparse
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import torchvision.transforms as transforms

import utils
import styleGAN
import dataset

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    parser = argparse.ArgumentParser(description='style gan')
    parser.add_argument("--yaml_file", type=str, default='./config.yaml')
    args = parser.parse_args()

    config = utils.load_yaml(args.yaml_file)

    style_img_path = config['dataset']['style_img']
    content_img_path = config['dataset']['content_img']

    ImageDataSet = dataset.ImageDataSet(device)

    style_img = ImageDataSet.image_loader(style_img_path)
    content_img = ImageDataSet.image_loader(content_img_path)

    plt.figure()
    utils.imshow(style_img, title='Style Image')
    plt.figure()
    utils.imshow(content_img, title='Content Image')

    num_steps = config['model']['num_steps']
    style_weight = config['model']['style_weight']
    content_weight = config['model']['content_weight']

    model = styleGAN.StyleGAN(style_img, content_img, device)
    output = model.run_style_transfer(num_steps=num_steps,
                                      style_weight=style_weight,
                                      content_weight=content_weight)

    plt.figure()
    utils.imshow(output, title='Output Image')

    # sphinx_gallery_thumbnail_number = 4
    plt.ioff()
    plt.show()

    save_dir = config['save_config']['save_dir']
    quality = config['save_config']['quality']
    file_name = config['save_config']['file_name']
    save_file_name = os.path.join(save_dir, file_name)

    utils.imsave(output, save_file_name)


if __name__ == "__main__":
    main()