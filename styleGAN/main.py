from __future__ import print_function
import argparse
import torch
import matplotlib.pyplot as plt

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


    style_img_path = "./data/images/neural-style/picasso.jpg"
    content_img_path = "./data/images/neural-style/dancing.jpg"

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



if __name__ == "__main__":
    main()