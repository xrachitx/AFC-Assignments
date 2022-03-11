from importlib.resources import path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import cv2 
from torch.autograd import Variable
from data_loader import LoadData
from torch.utils.data import Dataset, DataLoader
from model import Model
from tqdm import tqdm
import os
import argparse
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score


def parse_args():
    parser = argparse.ArgumentParser(description='Train ')
    parser.add_argument('--batch', default=20, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--checkpoints', default=5, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--rootDir', default=".", type=str)
    parser.add_argument('--files', default="train.csv", type=str)
    parser.add_argument('--testFile', default="test.csv", type=str)
    parser.add_argument('--ckptFolder', default="Checkpoints", type=str)
    parser.add_argument('--device', default="cuda", type=str)
    parser.add_argument('--logfile', default="log.csv", type=str)
    parser.add_argument('--freeze_encoder', default=False, action='store_true')
    parser.add_argument('--mtl', default=False, action='store_true')
    parser.add_argument('--gradRev', default=False, action='store_true')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    epochs = args.epochs
    rootDir =args.rootDir
    files = args.files
    lr = args.lr
    device = args.device
    freeze_encoder=args.freeze_encoder
    checkpoints = args.checkpoints
    batch_size = args.batch
    mtl = args.mtl
    gradRev = args.gradRev
    test_file = args.testFile
    log = args.logfile
    ckptFolder = args.ckptFolder

    if not (mtl or gradRev):
        ttype = "Baseline"
    elif mtl and not gradRev:
        ttype = "MTL"
    elif gradRev and not mtl:
        ttype = "Gradient Reversal"
    print(f"Training type: {ttype}")

    try:
        os.makedirs(ckptFolder)
    except:
        print("Checkpoint Folder Exists")


    td = LoadData(files, rootDir)
    train_dataloader = DataLoader(td,batch_size=batch_size)
    
    td = LoadData(test_file, rootDir)
    test_dataloader = DataLoader(td,batch_size=35)
    test_dataloader = list(test_dataloader)
    
    model = Model(device,gradRev,mtl,freeze_encoder)
    print(model)
    for params in model.parameters():
        params.requires_grad = True

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(),lr = lr)

    for epoch in tqdm(range(epochs)):
        loss_arr = []
        # print(f"Epoch: {epoch}-------Starting:")
        for i, (img,emotion,source_target) in enumerate(train_dataloader,0):

            img = img.to(device)
            emotion = emotion.to(device)
            source_target = source_target.to(device)
            model = model.to(device)

            if mtl or gradRev:
                pred_emotion,pred_classes = model(img)
                # print(pred_emotion,pred_classes)
                loss_emotion = criterion(pred_emotion,emotion)
                loss_class = criterion(pred_classes,source_target)
                loss = loss_emotion+loss_class
            else:
                pred_emotion = model(img)
                loss_emotion = criterion(pred_emotion,emotion)
                loss = loss_emotion
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_arr.append(loss.item())
#             exit()
        (img,emotion,source_target) = test_dataloader[0]
        img = img.cuda()
        emotion = emotion.cuda()
        if mtl or gradRev:
            pred_emotion,_ = model(img)
        else:
            pred_emotion = model(img)
        f1 = f1_score(emotion.cpu().numpy(),torch.argmax(pred_emotion,1).cpu().numpy(),labels=[0,1,2,3,4,5,6],average="micro")
        acc = accuracy_score(emotion.cpu().numpy(),torch.argmax(pred_emotion,1).cpu().numpy())
        print(f"Epoch: {epoch}-------Loss: {np.mean(loss_arr)}-------Test F1: {f1}-------Test Accuracy: {acc}")
        file = open(f"./{ckptFolder}/{log}",'a+')
        file.write(f"{epoch},{np.mean(loss_arr)},{f1},{acc}\n")
        file.close()

        if (epoch+1) % checkpoints == 0:
            path = f"./{ckptFolder}/model_{epoch}.pth"
            torch.save(model.state_dict(), path)

    path = f"./{ckptFolder}/model_{epochs}.pth"
    torch.save(model.state_dict(), path)
