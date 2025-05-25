# datasets/cassava_dataset.py
import os
import pandas as pd
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import numpy as np


class CassavaDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        label = int(self.data.iloc[idx, 1])
        #if self.transform:
        #    image = self.transform(image)
        if self.transform:
            image = np.array(image)  # PIL → NumPy (Albumentations 需要)
            image = self.transform(image=image)["image"]

        return image, label