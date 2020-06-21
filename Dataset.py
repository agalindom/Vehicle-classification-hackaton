import albumentations as A
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import joblib
import torch
from torch.utils.data import Dataset

def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec

class VehicleTrainDataset(Dataset):
    def __init__(self, dataframe, image_dir = None, transforms = None):
        super().__init__()
        
        self.df = dataframe
        self.image_dir = image_dir
        df = dataframe
        self.image_ids = df.image_names.values
        self.transforms = transforms
        
       
    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        image = Image.open(f'{self.image_dir}/{image_id}')
        image = image.convert('RGB')
        image = np.array(image)
        
        target = self.df["emergency_or_not"].iloc[index]
        
        sample = {
            "image" : image
        }
        
        if self.transforms:
            sample = self.transforms(**sample)
            image = sample['image']
        
        image = np.transpose(image, (2,0,1)).astype(np.float32)
        
        return {
            "image": torch.as_tensor(image, dtype = torch.float),
            "targets": torch.as_tensor(target, dtype = torch.long)
        }
    
    def __len__(self):
        return len(self.image_ids)
        

class VehicleTrainDataset2(Dataset):
    def __init__(self, dataframe, labels, image_dir = None, transforms = None):
        super().__init__()
        
        self.df = dataframe
        self.image_dir = image_dir
        df = dataframe
        self.image_ids = df.image_names.values
        self.transforms = transforms
        self.labels = labels
        
       
    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        image = Image.open(f'{self.image_dir}/{image_id}')
        image = image.convert('RGB')
        image = np.array(image)
        
        label = self.labels[index]
        target = onehot(2, label)
#         target = self.df["emergency_or_not"].iloc[index]
        
        sample = {
            "image" : image
        }
        
        if self.transforms:
            sample = self.transforms(**sample)
            image = sample['image']
        
        image = np.transpose(image, (2,0,1)).astype(np.float32)
        
        return {
            "image": torch.as_tensor(image, dtype = torch.float),
            "targets": torch.as_tensor(target, dtype = torch.long)
        }
    
    def __len__(self):
        return len(self.image_ids)