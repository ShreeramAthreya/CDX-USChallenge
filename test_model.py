import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.utils import save_image
from torchvision.transforms import ToTensor, ToPILImage

import os
from PIL import Image
from collections import OrderedDict
from GnD import Generator

def remove_module_from_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        name = ""
        if key.startswith('module.'):
            name = key[7:]
        else:
            name = key
        new_state_dict[name] = value
    return new_state_dict

def testModel(model, device, test_dir, result_dir):
    transform = transforms.Compose([transforms.ToTensor()])
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Do not calculate gradients
        for filename in os.listdir(test_dir):
            img_path = os.path.join(test_dir, filename)
            img = Image.open(img_path).convert('L')
            img = transform(img).unsqueeze(0).to(device)  # Convert to tensor and add batch dimension
            output = model(img).squeeze(0).cpu()  # Remove batch dimension and move to cpu
            output = (output + 1) / 2.0  # Rescale to 0-1
            output = output[0] * 255  # Rescale to 0-255
            output = output.numpy().astype('uint8') 
            output = Image.fromarray(output)
            output.save(os.path.join(result_dir, filename))

def testing(DEVICE):
    GeneratorH = Generator().to(DEVICE)
    state_dict = torch.load("genH.pth.tar", map_location=DEVICE)
    state_dict = remove_module_from_state_dict(state_dict["state_dict"])
    GeneratorH.load_state_dict(state_dict)
    test_dir = 'test'
    result_dir = 'results'
    os.makedirs(result_dir, exist_ok=True)
    testModel(GeneratorH, DEVICE, test_dir, result_dir)
