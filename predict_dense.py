import torch
# from utils.helpers import *
import warnings
from PIL import Image
from torchvision import transforms
# from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
import torchvision
from torchvision import models
import re
from torch.autograd import Variable


def image_transform(imagepath):
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Lambda(lambda x: x.repeat(3,1,1)),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    image = Image.open(imagepath)
    
    imagetensor = test_transforms(image)
    print(imagetensor.size())
    return imagetensor


def load_model(path):
    try:
        checkpoint = torch.load(path, map_location='cpu')
    except Exception as err:
        print(err)
        return None
    model = models.densenet121(pretrained=False)
    model.classifier = nn.Sequential(nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Dropout(0.1),
                                 nn.Linear(256, 2),
                                 nn.LogSoftmax(dim=1))
    model.parameters = checkpoint['parameters']
    model.load_state_dict(checkpoint['state_dict'])
    return model


def predict(imagepath, verbose=False):
    if not verbose:
        warnings.filterwarnings('ignore')
    model_path = './models/densenet.pth'

    model = load_model(model_path)
    model.eval()
    # summary(model, input_size=(3,244,244))
    if verbose:
        print("Model Loaded..")
    image = image_transform(imagepath)
    image1 = image[None, :, :, :]
    ps = torch.exp(model(image1))
    topconf, topclass = ps.topk(1, dim=1)
    if topclass.item() == 1:
        return {'class': 'Pneumonia', 'confidence': str(topconf.item())}
    else:
        return {'class': 'Normal', 'confidence': str(topconf.item())}



print(predict('data/normal4.jpeg'))
print(predict('data/normal2.jpeg'))
print(predict('data/normal1.jpeg'))
print(predict('data/normal5.jpeg'))
print(predict('data/pneumonia4.jpeg'))
print(predict('data/pneumonia5.png'))
print(predict('data/pneumonia1.jpeg'))
print(predict('data/pneumonia9.jpg'))
print(predict('data/pneumonia12.jpeg'))