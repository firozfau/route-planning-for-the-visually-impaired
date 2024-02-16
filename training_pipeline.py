import os
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
import albumentations as A
import matplotlib.pyplot as plt
from torchvision import transforms
from albumentations.pytorch import ToTensorV2
from torch.optim import lr_scheduler
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset

mapillary = 'Mapillary-Vistas-1000-sidewalks'
mapillary_labels = 'Mapillary_converted_masks'

Epochs = 100
learning_rate = 0.0001
encoder = 'resnet34'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 16
img_size = (256, 512)

void_classes = [1, 2, 3, 5, 6, 7, 4, 8, 9, 10, 11, 12, 13, 15, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33,
                34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 50, 52, 54, 57, 60, 61, 63, 64, 65, 28, 21, 62, 55, 59, 58]

target_classes = [66, 14, 16, 49, 51, 53, 56, 20]
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
        mask[mask == void_class] = 66
    for valid_class in target_classes:
        mask[mask == valid_class] = class_label_mapping[valid_class]
    return mask


def decode_segmap(segmentation_map):
    rgb = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8)
    for label, color in label_colors.items():
        rgb[segmentation_map == label] = color
    return rgb / 255.0


train_images = f'{mapillary}/training/images'
val_images = f'{mapillary}/validation/images'
train_masks = f'{mapillary_labels}/training'
val_masks = f'{mapillary_labels}/validation'

new_size = (256, 512)

transform = A.Compose([
    A.Resize(256, 512),
    A.HorizontalFlip(),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


class MapillaryDataset(Dataset):
    def __init__(self, image_folder, mask_folder, transform=None):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.transform = transform

        self.image_files = os.listdir(image_folder)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_folder, img_name)
        mask_name = img_name.replace('.jpg', '.png')
        mask_path = os.path.join(self.mask_folder, mask_name)

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            transformed = transform(image=np.array(image), mask=np.array(mask))
            return transformed['image'], transformed['mask']

        return image, mask


train_dataset = MapillaryDataset(image_folder=train_images,
                                 mask_folder=train_masks, transform=transform)

val_dataset = MapillaryDataset(image_folder=val_images,
                               mask_folder=val_masks, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


class SegmentationModel(nn.Module):
    def __init__(self, encoder='resnet34', num_classes=8):
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
model.load_state_dict(torch.load('base_model.pt', map_location=torch.device(device)))
model.layer.segmentation_head[0] = nn.Conv2d(model.layer.segmentation_head[0].in_channels, num_classes, kernel_size=1)
model.to(device);


def train_fn(data_loader, model, optimizer):
    model.train()
    total_loss = 0.0
    for images, masks in tqdm(data_loader):
        images = images.to(device)
        segment = masks.to(device)

        optimizer.zero_grad()
        out = model(images)
        segment = encode_segmap(segment.clone())
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


criterion = smp.losses.DiceLoss(mode='multiclass')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)

best_valid_loss = np.Inf
train_losses = []
valid_losses = []

checkpoint_folder = "mapillary_checkpoints"
os.makedirs(checkpoint_folder, exist_ok=True)

log_file_path = "mapillary_training_log.txt"
checkpoint_path = os.path.join(checkpoint_folder, 'mapillary_checkpoint.pt')

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
            torch.save(model.state_dict(), 'fine_tuned_model.pt')
            print('Model Saved!')
            best_valid_loss = valid_loss

        log_file.write(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}\n")

        print('\n')

print(f"Training log saved to {log_file_path}")

plt.style.use("ggplot")
plt.figure()
plt.plot(train_losses, label="train_loss")
plt.plot(valid_losses, label="valid_loss")
plt.title("Training Loss on Mapillary Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig('finetuned_loss_fig.png')
