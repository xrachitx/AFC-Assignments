import os
import random
import csv

base = "data/"
folders = ["Source"]


train = []
test = []
for folder in folders:
    emotions = os.listdir(f"{base}{folder}/")
    # fin.append([])
    for emotion in emotions:
        photos = os.listdir(f"{base}{folder}/{emotion}/")
        photos = [[f"{base}{folder}/{emotion}/{i}",folder,emotion] for i in photos]
        photos_target = os.listdir(f"{base}Target/{emotion}/")
        photos_target = [[f"{base}Target/{emotion}/{i}","Target",emotion] for i in photos_target]
        random.shuffle(photos_target)

        train +=  (photos[:] + photos_target[:5])
        test += photos_target[5:]

# print(train,test)

print(len(train),len(test)) 

with open("train.csv", 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(train)     

with open("test.csv", 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(test)