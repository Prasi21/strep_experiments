from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor




def Imageloader(inputpath,outputpath,start,max,verbose=0):
    count=0
    for dirname, _, filenames in os.walk(inputpath):
        for filename in filenames[start:max]:
            count=count+1
            if(verbose > 0):
                print(os.path.join(dirname, filename))
            image = Image.open(os.path.join(dirname, filename))
            transform = transforms.Compose([
                #transforms.PILToTensor(),
                transforms.Resize([224,224]),
                #transforms.functional.to_pil_image(image)
                ])
            image = transform(image)
            image.convert('RGB').save(f'{outputpath}/{count+start}.jpeg', format='jpeg')
            #torchvision.utils.save_image(tensor=image,fp=f'{outputpath}/{count}')
            if(verbose > 0):
                print(image)
    return count


root = "./datasets"


os.makedirs("./datasets/Original/Testfolder/Class0", exist_ok=True)
os.makedirs("./datasets/Original/Testfolder/Class1", exist_ok=True)
os.makedirs("./datasets/Original/Valfolder/Class0", exist_ok=True)
os.makedirs("./datasets/Original/Valfolder/Class1", exist_ok=True)
os.makedirs("./datasets/Original/Trainfolder/Class0", exist_ok=True)
os.makedirs("./datasets/Original/Trainfolder/Class1", exist_ok=True)


Imageloader(os.path.join(root,"raw_data","no_pharyngitis"),os.path.join(root, "Original", "Valfolder","Class0"),0,21)
Imageloader(os.path.join(root,"raw_data","no_pharyngitis"),os.path.join(root, "Original", "Testfolder","Class0"),21,42)
Imageloader(os.path.join(root,"raw_data","no_pharyngitis"),os.path.join(root, "Original", "Trainfolder","Class0"),42,208)

Imageloader(os.path.join(root,"raw_data","pharyngitis"),os.path.join(root, "Original", "Valfolder","Class1"),0,34)
Imageloader(os.path.join(root,"raw_data","pharyngitis"),os.path.join(root, "Original", "Testfolder","Class1"),34,68)
Imageloader(os.path.join(root,"raw_data","pharyngitis"),os.path.join(root, "Original", "Trainfolder","Class1"),68,339)


# train_data = torchvision.datasets.ImageFolder(root= os.path.join(root, "Trainfolder"), transform=ToTensor())
# train_data_loader = torch.utils.data.DataLoader(train_data,
#                                                 batch_size=21,
#                                                 shuffle=True,)

# val_data = torchvision.datasets.ImageFolder(root= os.path.join(root, "Valfolder"), transform=ToTensor())
# val_data_loader = torch.utils.data.DataLoader(val_data,
#                                                 batch_size=21,
#                                                 shuffle=True,)

# test_data = torchvision.datasets.ImageFolder(root= os.path.join(root, "Testfolder"), transform=ToTensor())
# test_data_loader = torch.utils.data.DataLoader(train_data,
#                                                 batch_size=21,
#                                                 shuffle=True,)