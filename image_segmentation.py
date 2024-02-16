import os
import glob
import torch
import shutil
import zipfile
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
import albumentations as A
import torch.optim as optim
from typing import Any, Tuple
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from torch.optim import lr_scheduler
import segmentation_models_pytorch as smp
from torchvision.datasets import Cityscapes
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from segmentation_models_pytorch.losses import DiceLoss

cityscapes = 'trainvaltest'
gtFine_source_folder = os.path.join(cityscapes, 'gtFine_trainvaltest', 'gtFine')
leftImg8bit_source_folder = os.path.join(cityscapes, 'leftImg8bit_trainvaltest', 'leftImg8bit')
gtFine_target_folder = os.path.join(cityscapes, 'gtFine')
leftImg8bit_target_folder = os.path.join(cityscapes, 'leftImg8bit')
try:
    shutil.move(gtFine_source_folder, gtFine_target_folder)
    shutil.move(leftImg8bit_source_folder, leftImg8bit_target_folder)
except Exception as e:
    pass

Epochs = 100
learning_rate = 0.001
encoder = 'resnet34'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 8
img_size = (256, 512)


ignore_index = 255
void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1, 11, 12, 13, 17, 21, 22, 23, 25, 27, 28, 31, 32, ]
target_classes = [ignore_index, 7, 8, 19, 20, 24, 26, 33]
class_labels = ['unlabeled', 'road', 'sidewalk', 'traffic_light', 'traffic_sign', 'person', 'car', 'bicycle']
class_label_mapping = dict(zip(target_classes, range(len(target_classes))))
num_classes = len(target_classes)
colors = [[0, 0, 0],
          [128, 64, 128],
          [244, 35, 232],
          [250, 170, 30],
          [220, 220, 0],
          [220, 20, 60],
          [0, 0, 142],
          [119, 11, 32],
          ]

label_colors = dict(zip(range(num_classes), colors))


def encode_segmap(mask):
    for void_class in void_classes:
        mask[mask == void_class] = ignore_index
    for valid_class in target_classes:
        mask[mask == valid_class] = class_label_mapping[valid_class]
    return mask


def decode_segmap(segmentation_map):
    rgb = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8)
    for label, color in label_colors.items():
        rgb[segmentation_map == label] = color
    return rgb / 255.0


""" Data Transformation"""

transform = A.Compose([
    A.Resize(256, 512),
    A.HorizontalFlip(),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

""" Custom Dataset Class """


class CustomClass(Cityscapes):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = Image.open(self.images[index]).convert('RGB')
        targets = [self._load_json(target) if t == 'polygon' else Image.open(target)
                   for t, target in zip(self.target_type, self.targets[index])]
        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transforms:
            transformed = transform(image=np.array(image), mask=np.array(target))
            return transformed['image'], transformed['mask']

        return image, target

    def __len__(self):
        return len(self.images)


trainset = CustomClass(cityscapes, split='train', mode='fine', target_type='semantic', transforms=transform)
validset = CustomClass(cityscapes, split='val', mode='fine', target_type='semantic', transforms=transform)

train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(validset, batch_size=batch_size, shuffle=False)

""" Segmentation Model """


class SegmentationModel(nn.Module):
    def __init__(self):
        super(SegmentationModel, self).__init__()
        self.layer = smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            activation=None
        )
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.layer(x)


model = SegmentationModel()
model.to(device);

""" Training and Evaluation Functions """


def train_fn(data_loader, model, optimizer):
    model.train()
    total_loss = 0.0
    for images, masks in tqdm(data_loader):
        images = images.to(device)
        segment = masks.to(device)

        optimizer.zero_grad()
        out = model(images)
        segment = encode_segmap(segment)
        loss = criterion(out, segment.long())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


def eval_fn(data_loader, model):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, masks in tqdm(data_loader):
            images = images.to(device)
            segment = masks.to(device)

            out = model(images)
            segment = encode_segmap(segment)
            loss = criterion(out, segment.long())

            total_loss += loss.item()

    return total_loss / len(data_loader)


""" Training Loop """

checkpoint_folder = "cityscapes_checkpoints"
os.makedirs(checkpoint_folder, exist_ok=True)

criterion = smp.losses.DiceLoss(mode='multiclass')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

best_valid_loss = np.Inf
train_losses = []
valid_losses = []

log_file_path = "cityscapes_training_log.txt"
checkpoint_path = os.path.join(checkpoint_folder, 'cityscapes_checkpoint.pt')

start_epoch = 1
'''Checkpoint'''
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    best_valid_loss = checkpoint['best_valid_loss']
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resumed training from epoch {start_epoch}")

with open(log_file_path, "w") as log_file:
    for epoch in range(start_epoch, Epochs + 1):
        print(f"Epoch: {epoch}")

        train_loss = train_fn(train_loader, model, optimizer)
        valid_loss = eval_fn(val_loader, model)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(f"Train Loss: {train_loss:.4f} Valid Loss: {valid_loss:.4f}")

        if valid_loss < best_valid_loss:
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_valid_loss': valid_loss
            }, checkpoint_path)
            torch.save(model.state_dict(), 'base_model.pt')
            print('Model Saved!')
            best_valid_loss = valid_loss

        log_file.write(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}\n")

        print('\n')

print(f"Training log saved to {log_file_path}")

""" Plot the training loss """

plt.style.use("ggplot")
plt.figure()
plt.plot(train_losses, label="train_loss")
plt.plot(valid_losses, label="valid_loss")
plt.title("Training Loss on Cityscapes Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig('base_loss_fig.png')
