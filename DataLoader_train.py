#DataLoader_train
import pandas as pd
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
#from torchvision.transforms import transforms
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import PIL
import cv2
from sklearn.model_selection import train_test_split


#batch_size = 32

# define transforms
affine_transform = torchvision.transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2))
#transforms.Lambda(gaussian_blur),

transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomApply([affine_transform], p=0.5),
                    transforms.CenterCrop(90),
                    #transforms.RandomCrop((80,80)),#
                    #transforms.RandomRotation(10, fill=(0,)),
                    transforms.Resize(28),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                    #transforms.RandomErasing(p=0.5, scale=(0.02, 0.10)),
                ])



# define transforms
affine_transform = torchvision.transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2))
#transforms.Lambda(gaussian_blur),

transform1 = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomApply([affine_transform], p=0.2),
                    transforms.CenterCrop(100),
                    #transforms.RandomResizedCrop(150, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333)),
                    #transforms.RandomHorizontalFlip(p=0.5),#
                    #transforms.RandomVerticalFlip(p=0.5),#
                    #transforms.RandomCrop((90,90)),
                    #transforms.RandomRotation(10, fill=(0,)),
                    #transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0)),
                    transforms.Resize(28),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                    #transforms.RandomErasing(p=0.5, scale=(0.02, 0.10)),
                ])


transform_val = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(28),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ])

# custom dataset
class Data(Dataset):
    def __init__(self, data, labels=None, transform=None, transform1=None):
        self.images = data
        self.labels = labels
        self.transforms = transform
        self.transforms1 = transform1

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image0 = self.images[index]
        image0 = np.asarray(image0).astype('uint8').reshape((105, 105, 1))
        label0 = self.labels[index]

        if self.transforms is not None:
            n = np.random.random()
            if n > 0.5: 
              image0 = self.transforms(image0)
            else:
              image0 = self.transforms1(image0)
            
        return image0, label0

        
class Data12345(Dataset):
    def __init__(self, data, labels=None, transform=None, transform1=None):
        self.images = data
        self.labels = labels
        self.transforms = transform
        self.transforms1 = transform1

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image1_index = np.random.randint(0, len(self.images))
        image2_index = np.random.randint(0, len(self.images))

        image0 = self.images[index]
        image0 = np.asarray(image0).astype('uint8').reshape((105, 105, 1))
        label0 = self.labels[index]
        
        for count, i in enumerate(self.labels):
            if i == label0:
                break
        
        image1 = self.images[count]
        image1 = np.asarray(image1).astype('uint8').reshape((105, 105, 1))
        label1 = self.labels[count]
    
        image2 = self.images[image2_index]
        image2 = np.asarray(image2).astype('uint8').reshape((105, 105, 1))
        label2 = self.labels[image2_index]

        if label0 == label2:
            image2 = self.images[image1_index]
            image2 = np.asarray(image2).astype('uint8').reshape((105, 105, 1))
            label2 = self.labels[image1_index]


        def rotated_imgs(rot_class):
              rotated_ = [
                        self.transforms1(image1),
                        self.transforms(rotate_img(image1,  90)),
                        self.transforms1(rotate_img(image1, 180)),
                        self.transforms(rotate_img(image1, 270))
                    ]
              return rotated_[rot_class]

        rot_class = np.random.randint(4)
        
        if self.transforms is not None:
            n = np.random.random()
            if n > 0.5: 
              image0 = self.transforms(image0)
              image1 = self.transforms(image1)
              image2 = self.transforms(image2)

            else:
              image0 = self.transforms1(image0)
              image1 = self.transforms1(image1)
              image2 = self.transforms1(image2)
        
        #rotated_imgs(rot_class)
        return image0, label0, image1, label1, image2, label2

        


def load_dataset(batch_size):
    # omniglot data
    data_dir = 'data_omniglot/'

    # Data Loader Omniglot
    training_data = np.load(data_dir + '/training_data.npy',allow_pickle=True)
    np.random.shuffle(training_data)
    print('Training_data', training_data.shape)

    X_train = [i[0] for i in training_data]
    y_train = [i[1] for i in training_data]


    # lets reserve 10% of our data for validation
    #VAL_PCT = 0.10
    #val_size = int(len(X_train)*VAL_PCT)
    #print(val_size)

    # train
    #train_X = X_train[:-val_size]
    #train_y = y_train[:-val_size]
    # val
    #val_X = X_train[-val_size:]
    #val_y = y_train[-val_size:]

    train_X, val_X, train_y, val_y = train_test_split(X_train, y_train, test_size=0.10, random_state=42)

    train_data = Data(train_X, train_y, transform, transform1)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)


    val_data = Data(val_X, val_y, transform_val, transform_val)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False) 

    return train_loader, val_loader