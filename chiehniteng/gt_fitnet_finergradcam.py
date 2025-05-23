import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnext50_32x4d
# from torch.utils.tensorboard import SummaryWriter

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
import wandb

TRAIN_PATH = '../input/cassava-leaf-disease-classification/train_images'
TEST_PATH = '../input/cassava-leaf-disease-classification/test_images'
OUTPUT_DIR = './'

class CustomTarget:
    def __init__(self, gt_class):
        self.gt_class = gt_class

    def __call__(self, model_output):
        if self.gt_class == 4:
            return model_output[4]
        else:
            return model_output[self.gt_class] - 0.6 * model_output[4]


class CFG:
    print_freq=1000
    num_workers = 4
    model_name = 'gt_fitnet_finergradcam'
    size = 512
    epochs = 30
    factor = 0.2
    patience = 5
    eps = 1e-6
    lr = 1e-4
    min_lr = 1e-6
    batch_size = 4
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
        self.conv1 = self.backbone.conv1
        self.bn1 = self.backbone.bn1
        self.relu = self.backbone.relu
        self.maxpool = self.backbone.maxpool
        self.layer1 = self.backbone.layer1
        self.layer2 = self.backbone.layer2
        self.layer3 = self.backbone.layer3
        self.layer4 = self.backbone.layer4
        self.avgpool = self.backbone.avgpool
        self.fc = self.backbone.fc

    def forward(self, x, return_feature=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        feat_layer2 = self.layer2(x)
        feat_layer3 = self.layer3(feat_layer2)
        x = self.layer4(feat_layer3)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)

        if return_feature:
            return logits, feat_layer3
        else:
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

    def forward(self, image, cam, return_feature=False):
        x = torch.cat([image, cam], dim=1)

        x = self.feature_extractor[0](x)  # conv1
        x = self.feature_extractor[1](x)  # bn1
        x = self.feature_extractor[2](x)  # relu
        x = self.feature_extractor[3](x)  # maxpool
        x = self.feature_extractor[4](x)  # layer1
        x = self.feature_extractor[5](x)  # layer2
        feat_layer3 = self.feature_extractor[6](x)  # layer3
        x = self.feature_extractor[7](feat_layer3)  # layer4
        x = self.feature_extractor[8](x)  # avgpool

        feat = self.flatten(x)
        class_logits = self.fc_class(feat)

        if return_feature:
            return class_logits, feat_layer3
        else:
            return class_logits


def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_k = k.replace("backbone.", "")
        new_state_dict[new_k] = v
    return new_state_dict


# Example training loop with validation and TensorBoard
if __name__ == '__main__':
    wandb.init(project="VRDL-Final",name=CFG.model_name)
    train = pd.read_csv('../input/cassava-leaf-disease-classification/train.csv')

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
    
    generator = resnext50_32x4d(weights='DEFAULT').cuda()
    generator.fc = nn.Linear(generator.fc.in_features, 5)
    state_dict = torch.load("./gan_v3_fold0_best_g_g.pth")['model']
    state_dict = remove_module_prefix(state_dict)
    generator.load_state_dict(state_dict)
    generator = generator.cuda()

    discriminator = Discriminator()
    state_dict = torch.load("./gt_gt_finergradcam_fold0_best_d_d.pth")['model']
    state_dict = remove_module_prefix(state_dict)
    discriminator.load_state_dict(state_dict)
    discriminator = discriminator.cuda()

    student = Generator().cuda()
    optimizer_S = torch.optim.Adam(student.parameters(),
                                 lr=1e-4,
                                 weight_decay=1e-6,
                                 amsgrad=False)
    scheduler_S = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_S,
        mode='min',
        factor=0.2,
        patience=5,
        verbose=True,
        eps=1e-6)
    optimizer_S_stage1 = torch.optim.Adam(
        list(student.conv1.parameters()) +
        list(student.bn1.parameters()) +
        list(student.layer1.parameters()) +
        list(student.layer2.parameters()) +
        list(student.layer3.parameters()), 
        lr=1e-4,
        weight_decay=1e-6,
        amsgrad=False
    )

    criterion_class = nn.CrossEntropyLoss()
    criterion_judge = nn.BCEWithLogitsLoss()
    criterion_kl = nn.KLDivLoss(reduction="batchmean")
    alpha = 1.0  # hard label loss
    beta = 1.0   # soft label loss
    gamma = 0.1
    temperature = 4.0

    best_score = 0
    best_s_score = 0

    target_layers = [generator.layer4[-1]]
    cam = GradCAMPlusPlus(model=generator, target_layers=target_layers)

    for epoch in range(30):
        generator.eval()
        discriminator.eval()
        student.train()

        losses_s = AverageMeter()
        losses_s_class = AverageMeter()
        losses_s_kl = AverageMeter()
        losses_s_f = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        start = end = time.time()
        total_samples = 0
        total_correct = 0
        total_s_correct = 0
        total_d_correct = 0

        # === Training ===
        for step, (images, labels) in enumerate(train_loader):

            data_time.update(time.time() - end)

            images = images.cuda()
            labels = labels.cuda()
            batch_size = labels.size(0)

            # generator
            logits = generator(images)
            preds = logits.argmax(dim=1)
            correct = (preds == labels)

            cams = []
            for i in range(batch_size):
                input_tensor = images[i].unsqueeze(0)
                gt = labels[i].item()
                target = [CustomTarget(gt)]
                grayscale_cam = cam(input_tensor=input_tensor, targets=target)[0]
                cams.append(torch.from_numpy(grayscale_cam).unsqueeze(0))
            cams = torch.stack(cams)

            # discriminator
            # d_class_logits = discriminator(images, cams)
            d_class_logits, d_features = discriminator(images.cuda(), cams.cuda(), return_feature=True)
            d_preds = d_class_logits.argmax(dim=1)
            d_correct = (d_preds == labels)

            # student
            s_class_logits = student(images)
            s_class_logits, s_features = student(images, return_feature=True)
            s_preds = s_class_logits.argmax(dim=1)
            s_correct = (s_preds == labels)
            # class loss
            s_class_loss = criterion_class(s_class_logits, labels)
            # kl loss
            T = temperature
            soft_teacher = F.softmax(d_class_logits / T, dim=1).detach()
            soft_student = F.log_softmax(s_class_logits / T, dim=1)
            s_kl_loss = criterion_kl(soft_student, soft_teacher) * (T * T)
            # feature loss
            s_f_loss = F.mse_loss(s_features, d_features.detach())

            if epoch < 10:
                s_loss = s_f_loss
                losses_s.update(s_loss.item(), batch_size)
                losses_s_class.update(s_class_loss.item(), batch_size)
                losses_s_kl.update(s_kl_loss.item(), batch_size)
                losses_s_f.update(s_f_loss.item(), batch_size)
                optimizer_S_stage1.zero_grad()
                s_loss.backward()
                optimizer_S_stage1.step()
            else:
                s_loss = alpha * s_class_loss + beta * s_kl_loss + gamma * s_f_loss
                losses_s.update(s_loss.item(), batch_size)
                losses_s_class.update(s_class_loss.item(), batch_size)
                losses_s_kl.update(s_kl_loss.item(), batch_size)
                losses_s_f.update(s_f_loss.item(), batch_size)
                optimizer_S.zero_grad()
                s_loss.backward()
                optimizer_S.step()

            total_correct += correct.sum().item()
            total_d_correct += d_correct.sum().item()
            total_s_correct += s_correct.sum().item()
            batch_size = labels.size(0)
            total_samples += batch_size

            batch_time.update(time.time() - end)
            end = time.time()
            if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
                print('Epoch: [{0}][{1}/{2}] '
                    'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                    'Elapsed {remain:s} '
                    'Loss_s: {loss_s.val:.4f}({loss_s.avg:.4f}) '
                    'Loss_s_class: {losses_s_class.val:.4f}({losses_s_class.avg:.4f}) '
                    'Loss_s_kl: {losses_s_kl.val:.4f}({losses_s_kl.avg:.4f}) '
                    'Loss_s_f: {losses_s_f.val:.4f}({losses_s_f.avg:.4f}) '
                    .format(
                        epoch, step, len(train_loader), batch_time=batch_time,
                        data_time=data_time,
                        remain=timeSince(start, float(step+1)/len(train_loader)),
                        loss_s=losses_s,
                        losses_s_class=losses_s_class,
                        losses_s_kl=losses_s_kl,
                        losses_s_f=losses_s_f,
                    ))
            del images, labels, cams, logits, d_class_logits, d_features, s_class_logits, s_features
            torch.cuda.empty_cache()
        train_g_acc = total_correct/total_samples
        train_d_acc = total_d_correct/total_samples
        train_s_acc = total_s_correct/total_samples
        print(f"Epoch {epoch} Training | "
              f"G: {train_g_acc:.4f} | "
              f"D: {train_d_acc:.4f} | "
              f"S: {train_s_acc:.4f} | ")
        # writer.add_scalar("Train/Loss_S", s_loss.item(), epoch)
        # writer.add_scalar("Accuracy/Train_S", s_correct.float().mean().item(), epoch)

        wandb.log({
            "train_g_acc": train_g_acc,
            "train_d_acc": train_d_acc,
            "train_s_acc": train_s_acc,
            "train_loss": losses_s.avg,
            "train_class_loss": losses_s_class.avg,
            "train_kl_loss": losses_s_kl.avg,
            "train_f_loss": losses_s_f.avg,
            "epoch": epoch
        }, commit=False)

        
        # === Validation === (simulate validation set)
        generator.eval()
        discriminator.eval()
        student.eval()

        losses_s = AverageMeter()
        losses_s_class = AverageMeter()
        losses_s_kl = AverageMeter()
        losses_s_f = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        start = end = time.time()
        total_samples = 0
        total_correct = 0
        total_s_correct = 0
        total_d_correct = 0

        for step, (images, labels) in enumerate(valid_loader):
            data_time.update(time.time() - end)

            images = images.cuda()
            labels = labels.cuda()
            batch_size = labels.size(0)

            # generator
            logits = generator(images)
            preds = logits.argmax(dim=1)
            correct = (preds == labels)

            cams = []
            for i in range(batch_size):
                input_tensor = images[i].unsqueeze(0)
                gt = labels[i].item()
                target = [CustomTarget(gt)]
                grayscale_cam = cam(input_tensor=input_tensor, targets=target)[0]
                cams.append(torch.from_numpy(grayscale_cam).unsqueeze(0))
            cams = torch.stack(cams)
        
            with torch.no_grad():
                # discriminator
                d_class_logits, d_features = discriminator(images.cuda(), cams.cuda(), return_feature=True)
                d_preds = d_class_logits.argmax(dim=1)
                d_correct = (d_preds == labels)
                # student
                s_class_logits, s_features = student(images, return_feature=True)
                s_preds = s_class_logits.argmax(dim=1)
                s_correct = (s_preds == labels)

                # class loss
                s_class_loss = criterion_class(s_class_logits, labels)
                # kl loss
                T = temperature
                soft_teacher = F.softmax(d_class_logits / T, dim=1).detach()
                soft_student = F.log_softmax(s_class_logits / T, dim=1)
                s_kl_loss = criterion_kl(soft_student, soft_teacher) * (T * T)
                # feature loss
                s_f_loss = F.mse_loss(s_features.detach(), d_features.detach())

                if epoch < 10:
                    s_loss = s_f_loss
                else:
                    s_loss = alpha * s_class_loss + beta * s_kl_loss + gamma * s_f_loss

                losses_s.update(s_loss.item(), batch_size)
                losses_s_class.update(s_class_loss.item(), batch_size)
                losses_s_kl.update(s_kl_loss.item(), batch_size)
                losses_s_f.update(s_f_loss.item(), batch_size)

                total_correct += correct.sum().item()
                total_d_correct += d_correct.sum().item()
                total_s_correct += s_correct.sum().item()
                batch_size = labels.size(0)
                total_samples += batch_size

                batch_time.update(time.time() - end)
                end = time.time()
                if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
                    print('Epoch: [{0}][{1}/{2}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Elapsed {remain:s} '
                        'Loss_s: {loss_s.val:.4f}({loss_s.avg:.4f}) '
                        'Loss_s_class: {losses_s_class.val:.4f}({losses_s_class.avg:.4f}) '
                        'Loss_s_kl: {losses_s_kl.val:.4f}({losses_s_kl.avg:.4f}) '
                        'Loss_s_f: {losses_s_f.val:.4f}({losses_s_f.avg:.4f}) '
                        .format(
                            epoch, step, len(train_loader), batch_time=batch_time,
                            data_time=data_time,
                            remain=timeSince(start, float(step+1)/len(train_loader)),
                            loss_s=losses_s,
                            losses_s_class=losses_s_class,
                            losses_s_kl=losses_s_kl,
                            losses_s_f=losses_s_f,
                        ))
            del images, labels, cams, logits, d_class_logits, d_features, s_class_logits, s_features
            torch.cuda.empty_cache()

        val_acc = total_correct/total_samples
        val_d_acc = total_d_correct/total_samples
        val_s_acc = total_s_correct/total_samples
        print(f"Epoch {epoch} Val | "
            f"G: {val_acc:.4f} | "
            f"D: {val_d_acc:.4f} | "
            f"S: {val_s_acc:.4f} | ")

        wandb.log({
            "val_g_acc": val_acc,
            "val_d_acc": val_d_acc,
            "val_s_acc": val_s_acc,
            "val_loss": losses_s.avg,
            "val_class_loss": losses_s_class.avg,
            "val_kl_loss": losses_s_kl.avg,
            "val_f_loss": losses_s_f.avg,
            "epoch": epoch
        })

        scheduler_S.step(losses_s.avg)

        if val_s_acc > best_s_score:
            best_s_score = val_s_acc
            if epoch < 10:
                print(f'Epoch {epoch} - Save Best S Score: {best_s_score:.4f} Model')
                torch.save({'model': student.state_dict(), 'preds': preds}, OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_best_d_d_stage1.pth')
            else:
                print(f'Epoch {epoch} - Save Best S Score: {best_s_score:.4f} Model')
                torch.save({'model': student.state_dict(), 'preds': preds}, OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_best_d_d.pth')
