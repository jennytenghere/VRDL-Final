import os
import pandas as pd
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from albumentations import Compose, Normalize, HorizontalFlip, VerticalFlip
from albumentations.pytorch import ToTensorV2
from albumentations import (
    HorizontalFlip, VerticalFlip, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    GaussNoise, MotionBlur, MedianBlur, RandomResizedCrop,
    RandomBrightnessContrast, OneOf, Compose, Normalize, CoarseDropout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize, Affine
)


CFG = {
    'fold_num': 5,
    'seed': 719,
    'model_arch': 'vit_base_patch16_384',
    'img_size': 384,
    'epochs': 10,
    'train_bs': 10,
    'valid_bs': 10,
    'T_0': 10,
    'lr': 1e-4,
    'min_lr': 1e-4, 
    'smoothing' : 0.06,
    't1' : 0.8,
    't2' : 1.4,
    'warmup_factor' : 7,
    'warmup_epo' : 1,
    'num_workers': 4,#8
    'accum_iter': 2, # batch accumulation for backprop with effectively larger batch size
    'verbose_step': 1
}

def get_train_transforms():
    return Compose([
            RandomResizedCrop((CFG['img_size'], CFG['img_size'])),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            Affine(translate_percent=0.0625, scale=0.1, rotate=45, p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            CoarseDropout(p=0.5),
            CoarseDropout(p=0.5),
            ToTensorV2(p=1.0),
        ], p=1.)

def get_valid_transforms():
    return Compose([
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            CenterCrop(CFG['img_size'], CFG['img_size'], p=1.),
            Resize(CFG['img_size'], CFG['img_size']),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)

train_transforms = transforms.Compose([
        transforms.Resize((510, 510), Image.BILINEAR),
        transforms.RandomCrop((384, 384)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(45),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=0.1),
        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
])

val_transforms = transforms.Compose([
        transforms.Resize((510, 510), Image.BILINEAR),
        transforms.CenterCrop((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
])


class CassavaDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image_id'])
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
        label = int(row['label'])

        if self.transform:
            img = self.transform(image=img)['image']

        return idx, img, label
    

def build_loader(fold):
    data = pd.read_csv("../input/cassava-leaf-disease-classification/data.csv")

    # 切分
    train_df = data[data['fold'] != fold].copy()
    val_df = data[data['fold'] == fold].copy()

    # 設定路徑與 transform
    img_dir = '../input/cassava-leaf-disease-classification/train_images'
    train_dataset = CassavaDataset(train_df, img_dir, transform=get_train_transforms())
    val_dataset = CassavaDataset(val_df, img_dir, transform=get_valid_transforms())

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    return train_loader, val_loader

if __name__ == "__main__":
    train_loader, _ = build_loader(1)

    for image, label in train_loader:
        print(image.shape)
        print(label.shape)
        break
    
    

