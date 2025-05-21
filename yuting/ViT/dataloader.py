import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from fmix import sample_mask, make_low_freq_image, binarise_mask
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

def get_img(path):
    im_bgr = cv2.imread(path)
    im_rgb = im_bgr[:, :, ::-1]
    return im_rgb

def rand_bbox(size, lam):
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


class CassavaDataset(Dataset):
    def __init__(self, df, data_root, 
                 transforms=None, 
                 output_label=True, 
                 one_hot_label=False,
                 do_fmix=False, 
                 fmix_params={
                     'alpha': 1., 
                     'decay_power': 3., 
                     'shape': (CFG['img_size'], CFG['img_size']),
                     'max_soft': True, 
                     'reformulate': False
                 },
                 do_cutmix=False,
                 cutmix_params={
                     'alpha': 1,
                 }
                ):
        
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.data_root = data_root
        self.do_fmix = do_fmix
        self.fmix_params = fmix_params
        self.do_cutmix = do_cutmix
        self.cutmix_params = cutmix_params
        
        self.output_label = output_label
        self.one_hot_label = one_hot_label
        
        if output_label == True:
            self.labels = self.df['label'].values

            if one_hot_label is True:
                self.labels = np.eye(self.df['label'].max()+1)[self.labels]

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index: int):
        
        # get labels
        if self.output_label:
            target = self.labels[index]
          
        img  = get_img("{}/{}".format(self.data_root, self.df.loc[index]['image_id']))

        if self.transforms:
            img = self.transforms(image=img)['image']
        
        if self.do_fmix and np.random.uniform(0., 1., size=1)[0] > 0.5:
            with torch.no_grad():
                lam = np.clip(np.random.beta(self.fmix_params['alpha'], self.fmix_params['alpha']),0.6,0.7)
                
                # Make mask, get mean / std
                mask = make_low_freq_image(self.fmix_params['decay_power'], self.fmix_params['shape'])
                mask = binarise_mask(mask, lam, self.fmix_params['shape'], self.fmix_params['max_soft'])
    
                fmix_ix = np.random.choice(self.df.index, size=1)[0]
                fmix_img  = get_img("{}/{}".format(self.data_root, self.df.iloc[fmix_ix]['image_id']))

                if self.transforms:
                    fmix_img = self.transforms(image=fmix_img)['image']

                mask_torch = torch.from_numpy(mask)
                
                # mix image
                img = mask_torch*img+(1.-mask_torch)*fmix_img

                rate = mask.sum()/CFG['img_size']/CFG['img_size']
                target = rate*target + (1.-rate)*self.labels[fmix_ix]
       
        if self.do_cutmix and np.random.uniform(0., 1., size=1)[0] > 0.5:
            with torch.no_grad():
                cmix_ix = np.random.choice(self.df.index, size=1)[0]
                cmix_img  = get_img("{}/{}".format(self.data_root, self.df.iloc[cmix_ix]['image_id']))
                if self.transforms:
                    cmix_img = self.transforms(image=cmix_img)['image']
                    
                lam = np.clip(np.random.beta(self.cutmix_params['alpha'], self.cutmix_params['alpha']),0.3,0.4)
                bbx1, bby1, bbx2, bby2 = rand_bbox((CFG['img_size'], CFG['img_size']), lam)

                img[:, bbx1:bbx2, bby1:bby2] = cmix_img[:, bbx1:bbx2, bby1:bby2]

                rate = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (CFG['img_size'] * CFG['img_size']))
                target = rate*target + (1.-rate)*self.labels[cmix_ix]
                      
        if self.output_label == True:
            return img, target
        else:
            return img
        
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

def prepare_dataloader(train_, valid_, data_root='../input/cassava-leaf-disease-classification/train_images/'):   
   
    train_ds = CassavaDataset(train_, data_root, transforms=get_train_transforms(), output_label=True, one_hot_label=False, do_fmix=0, do_cutmix=0)
    valid_ds = CassavaDataset(valid_, data_root, transforms=get_valid_transforms(), output_label=True)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=32,
        shuffle=True,       
        num_workers=4,      
        # pin_memory=True     
    )
    val_loader = DataLoader(
        valid_ds,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        # pin_memory=True     
    )

    return train_loader, val_loader
if __name__ == "__main__":
    data = pd.read_csv('./data.csv')
    data = data.rename(columns={'target': 'label', 'img_name' : "image_id"})
    folds = 5
    for fold in range(folds):
        trn_idx = data[data['fold'] != fold]
        val_idx = data[data['fold'] == fold]
        train_dl, val_dl = prepare_dataloader(trn_idx, val_idx, data_root='./input/cassava-leaf-disease-classification/train_images/')
        for img, label in train_dl:
            print(img.shape)
            print(label.shape)
            break
        break