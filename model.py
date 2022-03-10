import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import cv2 
from torch.autograd import Variable
from data_loader import LoadData
from torch.utils.data import Dataset, DataLoader

class Model(nn.Module):
    def __init__(self,device,mtl=False,freeze_encoder=False):
        super().__init__()
        self.emotion_classes = 7
        self.num_classes = 2
        self.mtl = mtl
        vgg = models.vgg16(pretrained = True)
        if freeze_encoder:
            for param in vgg.parameters():
                param.requires_grad = False
        self.vgg16 = nn.ModuleList(list(vgg.features))
        self.fc_layer = nn.Sequential(
                nn.Linear(in_features=25088, out_features=1024),
                nn.ReLU()
            )
        self.emotion_output = nn.Sequential(
            nn.Linear(in_features=1024, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=self.emotion_classes)
        )
        self.class_out = nn.Sequential(
            nn.Linear(in_features=1024, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=self.num_classes)
        )
        self.softmax = nn.Softmax(dim=1)
        self.device = device


    def forward(self, x):
        for model in self.vgg16:
            x = model(x)
        x = torch.flatten(x,1)
        y = self.fc_layer(x)
        x = self.fc_layer(x)

        y = self.emotion_output(y)
        y = self.softmax(y)

        if self.mtl:

            x = self.class_out(x)
            x = self.softmax(x)

            # print(y.shape,x.shape)
            return y,x
        return y
if __name__ == "__main__":
    # rootDir ="./CoSkel+"
    # files = "./CoSkel+/train.csv"
    rootDir ="./"
    files = "train.csv"
    td = LoadData(files, rootDir)
    train_dataloader = DataLoader(td,batch_size=20)
    e = Model("cpu")
    print(e)
    # print(train_dataloader)
    for i, (data) in enumerate(train_dataloader,0):
    #    print(data[0].shape,data[1].shape)
       y,x = e(data[0])
       print(y)
       print(x)
       exit()