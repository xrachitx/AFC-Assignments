import os
import numpy as np
import pandas as pd
from torchvision import transforms
import skimage.io as io
import skimage
from torch.utils.data import Dataset, DataLoader
import torch
import cv2

emotion2label = {'Angry': 0,
                'Disgust': 1,
                'Fear': 2,
                'Happy': 3,
                'Neutral': 4,
                'Sad': 5,
                'Surprise': 6}

sourcetarget2label = {"Source": 0, "Target": 1}

class LoadData(Dataset):
    def __init__(self, fileNames, rootDir, transform=transforms.Compose([transforms.ToPILImage(),transforms.Resize((224, 224)), transforms.ToTensor()])):
        self.rootDir = rootDir
        self.transform = transform
        self.frame = pd.read_csv(fileNames, dtype=str, delimiter=',', header=None)
        print(self.frame)
    
    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):

        inputName = os.path.join(self.rootDir, self.frame.iloc[idx, 0])
        source_target =  sourcetarget2label[self.frame.iloc[idx, 1]]
        emotion =  emotion2label[self.frame.iloc[idx, 2]]
        inputImage = cv2.imread(inputName)
        inputImage = self.transform(inputImage)
        return inputImage,emotion,source_target

if __name__ == "__main__":
    rootDir ="./"
    files = "train.csv"
    # img_transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize((224, 224)), transforms.ToTensor()])
    td = LoadData(files, rootDir)
    train_dataloader = DataLoader(td,batch_size=20,shuffle=True)
    
    # print(train_dataloader)
    for i, (data) in enumerate(train_dataloader,0):
        print(data[0].shape,data[1],data[2])
        exit()
    # print(len(train_dataloader))