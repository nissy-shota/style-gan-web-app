from __future__ import print_function
import torch
from PIL import Image
import torchvision.transforms as transforms

class ImageDataSet():

    def __init__(self, device):
        self.device = device
        # desired size of the output image
        self.imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

        self.loader = transforms.Compose([
            transforms.Resize(self.imsize),  # scale imported image
            transforms.ToTensor()])  # transform it into a torch tensor


    def image_loader(self, image_name):

        image = Image.open(image_name)
        # fake batch dimension required to fit network's input dimensions
        image = self.loader(image).unsqueeze(0)
        return image.to(self.device, torch.float)
