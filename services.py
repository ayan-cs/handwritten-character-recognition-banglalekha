from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os

class DataGenerator(Dataset):
    def __init__(self, img_source, transform):
        self.transform = transform
        self.x = []
        self.y = []

        for label in os.listdir(img_source):
            all_imgs = os.listdir(os.path.join(img_source, label))
            for img in all_imgs:
                temp = Image.open(os.path.join(img_source, label, img))
                self.x.append(np.array(temp))
                self.y.append(int(label))
                temp.close()
        
        self.size = len(y)
        self.x = np.array(self.x)
        self.y = np.array(self.y)
    
    def __getitem__(self, i):
        img = self.x[i]
        if self.transform is not None:
            img = self.transform(img)
        label = self.y[i]
        return img, label
    
    def __len__(self, i):
        return self.size

class EarlyStopper :
    def __init__(self, patience = 5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False