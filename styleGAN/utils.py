import yaml
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

def load_yaml(yaml_file):
    """
    Load yaml file.
    """
    with open(yaml_file) as stream:
        param = yaml.safe_load(stream)
    return param

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    unloader = transforms.ToPILImage()
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated