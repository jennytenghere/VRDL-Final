import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnext50_32x4d
from torch.utils.tensorboard import SummaryWriter

from albumentations import (Compose, Normalize, Resize, RandomResizedCrop, HorizontalFlip, VerticalFlip, ShiftScaleRotate, Transpose)
from albumentations.pytorch import ToTensorV2
import cv2
import math
import pandas as pd
import time
from torch.utils.data import DataLoader, Dataset

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import numpy as np
from collections import OrderedDict


TRAIN_PATH = '../input/cassava-leaf-disease-classification/train_images'
TEST_PATH = '../input/cassava-leaf-disease-classification/test_images'
OUTPUT_DIR = './'

class CustomTarget:
    def __init__(self, pred_class):
        self.pred_class = pred_class

    def __call__(self, model_output):
        return model_output[self.pred_class]

class CFG:
    print_freq=300
    num_workers = 4
    model_name = 'gt_gt_gradcam'
    size = 512
    epochs = 30
    factor = 0.2
    patience = 5
    eps = 1e-6
    lr = 1e-4
    min_lr = 1e-6
    batch_size = 12
    weight_decay = 1e-6
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
    seed = 42
    target_size = 5
    target_col = 'label'
    n_fold = 5
    trn_fold = [1,2,3,4,5]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_transforms(*, data):
    if data == 'train':
        return Compose([
            RandomResizedCrop((CFG.size, CFG.size), scale=(0.4, 1.0)),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    
    elif data == 'valid':
       return Compose([
           Resize(CFG.size, CFG.size),
           Normalize(
               mean=[0.485, 0.456, 0.406],
               std=[0.229, 0.224, 0.225],
           ),
           ToTensorV2(),
       ])


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


class TrainDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.file_names = df['image_id'].values
        self.labels = df['label'].values
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = f'{TRAIN_PATH}/{file_name}'
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        label = torch.tensor(self.labels[idx]).long()
        return image, label


class Generator(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.backbone = resnext50_32x4d(weights='DEFAULT')
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        logits = self.backbone(x)
        return logits

    def get_cam(self, x, class_idx=None):
        features = []
        gradients = []

        def forward_hook(module, input, output):
            features.append(output)

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        handle_f = self.backbone.layer4.register_forward_hook(forward_hook)
        handle_b = self.backbone.layer4.register_full_backward_hook(backward_hook)

        logits = self.forward(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1)

        one_hot = torch.zeros_like(logits)
        for i in range(logits.size(0)):
            one_hot[i, class_idx[i]] = 1

        self.zero_grad()
        logits.backward(gradient=one_hot, retain_graph=True)

        cams = []
        for i in range(x.size(0)):
            grad = gradients[0][i]
            feat = features[0][i]
            weights = grad.mean(dim=(1, 2))
            cam = torch.sum(weights.view(-1, 1, 1) * feat, dim=0)
            cam = F.relu(cam)
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
            cams.append(cam.unsqueeze(0))

        cam_tensor = torch.stack(cams)
        cam_tensor = F.interpolate(cam_tensor, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        handle_f.remove()
        handle_b.remove()
        return cam_tensor.detach()

class Discriminator(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        base = resnext50_32x4d(weights='DEFAULT')
        base.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feature_extractor = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
            base.avgpool,
        )
        self.flatten = nn.Flatten()
        in_features = base.fc.in_features
        self.fc_class = nn.Linear(in_features, num_classes)

    def forward(self, image, cam):
        x = torch.cat([image, cam], dim=1)
        feat = self.feature_extractor(x)
        feat = self.flatten(feat)
        class_logits = self.fc_class(feat)
        return class_logits


def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_k = k.replace("backbone.", "")
        new_state_dict[new_k] = v
    return new_state_dict


# Example training loop with validation and TensorBoard
if __name__ == '__main__':
    train = pd.read_csv('../input/cassava-leaf-disease-classification/train.csv')
    # Split into folds for cross validation - we used the same split for all the models we trained!
    folds = train.merge(
        pd.read_csv("../validation_data.csv")[["image_id", "fold"]], on="image_id")
    
    fold = 0
    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)

    train_dataset = TrainDataset(train_folds, transform=get_transforms(data='train'))
    valid_dataset = TrainDataset(valid_folds, transform=get_transforms(data='valid'))

    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, 
                              shuffle=True, num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size, 
                              shuffle=False, num_workers=CFG.num_workers, pin_memory=True, drop_last=False)

    # load
    generator = resnext50_32x4d(weights='DEFAULT').cuda()
    generator.fc = nn.Linear(generator.fc.in_features, 5)
    state_dict = torch.load("./gan_v3_fold0_best_g_g.pth")['model']
    state_dict = remove_module_prefix(state_dict)
    generator.load_state_dict(state_dict)
    generator = generator.cuda()
    discriminator = Discriminator().cuda()

    optimizer_D = torch.optim.Adam(discriminator.parameters(),
                                 lr=1e-4,
                                 weight_decay=1e-6,
                                 amsgrad=False)
    scheduler_D = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_D,
        mode='min',
        factor=0.2,
        patience=5,
        verbose=True,
        eps=1e-6)

    criterion_class = nn.CrossEntropyLoss()
    criterion_judge = nn.BCEWithLogitsLoss()
    writer = SummaryWriter()
    best_score = 0
    best_d_score = 0

    target_layers = [generator.layer4[-1]]
    cam = GradCAMPlusPlus(model=generator, target_layers=target_layers)

    for epoch in range(20):
        generator.eval()
        discriminator.train()
        losses_d = AverageMeter()
        losses_d_class = AverageMeter()
        losses_d_judge = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        start = end = time.time()

        # === Training ===
        for step, (images, labels) in enumerate(train_loader):

            data_time.update(time.time() - end)

            images = images.cuda()
            labels = labels.cuda()
            batch_size = labels.size(0)
            
            logits = generator(images)
            preds = logits.argmax(dim=1)
            correct = (preds == labels)

            cams = []
            for i in range(batch_size):
                input_tensor = images[i].unsqueeze(0)
                gt = labels[i].item()
                target = [CustomTarget(gt)]
                grayscale_cam = cam(input_tensor=input_tensor, targets=target)[0]
                cams.append(grayscale_cam)

            cams = torch.from_numpy(np.stack(cams)).unsqueeze(1).float().cuda()

            d_class_logits = discriminator(images, cams)
            d_class_loss = criterion_class(d_class_logits, labels)
            judge_labels = (preds == labels).float().unsqueeze(1)
            d_loss = d_class_loss

            losses_d.update(d_loss.item(), batch_size)
            losses_d_class.update(d_class_loss.item(), batch_size)

            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

            batch_time.update(time.time() - end)
            end = time.time()
            if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
                print('Epoch: [{0}][{1}/{2}] '
                    'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                    'Elapsed {remain:s} '
                    'Loss_d: {loss_d.val:.4f}({loss_d.avg:.4f}) '
                    'Loss_d_class: {losses_d_class.val:.4f}({losses_d_class.avg:.4f}) '
                    
                    .format(
                    epoch+1, step, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss_d=losses_d,
                    losses_d_class=losses_d_class,
                    remain=timeSince(start, float(step+1)/len(train_loader)),
                    ))

        writer.add_scalar("Loss/Discriminator", d_loss.item(), epoch)
        writer.add_scalar("Accuracy/Train_Gen", correct.float().mean().item(), epoch)

        # === Validation === (simulate validation set)
        losses_g = AverageMeter()
        losses_d = AverageMeter()
        generator.eval()
        discriminator.eval()
        total_correct = 0
        total_d_correct = 0
        total_samples = 0
        
        for step, (val_images, val_labels) in enumerate(valid_loader):
            batch_size = val_labels.size(0)
            val_images = val_images.cuda()
            val_labels = val_labels.cuda()

            val_logits = generator(val_images)
            val_preds = val_logits.argmax(dim=1)
            val_correct = (val_preds == val_labels)

            val_cams = []
            for i in range(batch_size):
                input_tensor = val_images[i].unsqueeze(0)
                gt = val_labels[i].item()
                target = [CustomTarget(gt)]
                grayscale_cam = cam(input_tensor=input_tensor, targets=target)[0]
                val_cams.append(grayscale_cam)

            val_cams = torch.from_numpy(np.stack(val_cams)).unsqueeze(1).float().cuda()

            
            with torch.no_grad():
                val_g_loss = criterion_class(val_logits, val_labels)
                val_d_class_logits = discriminator(val_images, val_cams)
                val_d_class_loss = criterion_class(val_d_class_logits, val_labels)
                val_d_preds = val_d_class_logits.argmax(dim=1)
                val_judge_labels = (val_preds == val_labels).float().unsqueeze(1)
                val_d_loss = val_d_class_loss
                val_d_correct = (val_d_preds == val_labels)

            losses_g.update(val_g_loss.item(), batch_size)
            losses_d.update(val_d_loss.item(), batch_size)
            total_correct += val_correct.sum().item()
            total_d_correct += val_d_correct.sum().item()
            total_samples += batch_size

        writer.add_scalar("Val/Loss_Gen", val_g_loss.item(), epoch)
        writer.add_scalar("Val/Loss_Dis", val_d_loss.item(), epoch)
        writer.add_scalar("Accuracy/Val_Gen", val_correct.float().mean().item(), epoch)
            
        scheduler_D.step(losses_d.avg)
        epoch_val_acc = total_correct / total_samples
        epoch_val_d_acc = total_d_correct / total_samples
        print(f"Epoch {epoch} | Val G Accuracy: {epoch_val_acc:.4f} | Val D Accuracy: {epoch_val_d_acc:.4f}")
        print(f"Gen Loss: {losses_g.avg:.4f} | Dis Loss: {losses_d.avg:.4f}")

        if epoch_val_d_acc > best_d_score:
            best_d_score = epoch_val_d_acc
            print(f'Epoch {epoch+1} - Save Best D Score: {best_d_score:.4f} Model')
            torch.save({'model': discriminator.state_dict(), 'preds': preds}, OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_best_d_d.pth')
    writer.close()
