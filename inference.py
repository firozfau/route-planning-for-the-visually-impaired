import numpy as np
import os
import torch
import sys
import cv2
import torch.nn as nn
import segmentation_models_pytorch as smp
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

colors = [[0, 0, 0],
          [128, 64, 128],
          [244, 35, 232],
          [250, 170, 30],
          [220, 220, 0],
          [220, 20, 60],
          [0, 0, 142],
          [119, 11, 32],
          ]
num_classes = 8
label_colors = dict(zip(range(num_classes), colors))


# Define Segmentation Model
class SegmentationModel(nn.Module):
    def __init__(self):
        super(SegmentationModel, self).__init__()
        self.layer = smp.DeepLabV3Plus(
            encoder_name='resnet34',
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            activation=None
        )
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.layer(x)


# Decoding Segmentation Map
def decode_segmap(segmentation_map):
    rgb = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8)
    for label, color in label_colors.items():
        rgb[segmentation_map == label] = color
    return rgb / 255.0


def create_binary_mask(img, rgb_value):
    mask = np.all(img == np.array(rgb_value).reshape(1, 1, 3), axis=2).astype(np.uint8) * 255
    return mask


# Creating Binary Mask for Ground Truth Labels of Mapillary Dataset
def create_ground_truth_binary_mask(labels, rgb_value):
    print("\n****** Creating Ground Truth Binary Masks: ******")
    for filename in tqdm(os.listdir(labels), position=0, leave=True):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(labels, filename)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if image is not None:
                binary_mask = create_binary_mask(image, tuple(rgb_value))
                binary_mask = cv2.resize(binary_mask, (512, 256), interpolation=cv2.INTER_NEAREST)

                output_path = os.path.join(ground_truth_dir, filename)
                cv2.imwrite(output_path, binary_mask)

    print("Ground truth binary masks saved in:", ground_truth_dir, "\n")


if __name__ == '__main__':
    if len(sys.argv) != 7:
        print("Usage:# python inference.py path/to/model path/to/mapillaryImages path/to/mapillaryLabels "
              "OutputPath/to/ModelOutput OutputPath/to/binaryPredictions OutputPath/to/groundTruth")
        sys.exit(1)

    model_path = sys.argv[1]
    mapillary_images = sys.argv[2]
    mapillary_labels = sys.argv[3]
    model_output_folder = sys.argv[4]
    prediction_binary_dir = sys.argv[5]
    ground_truth_dir = sys.argv[6]
    rgb = [244, 35, 232]

    os.makedirs(model_output_folder, exist_ok=True)

    if not os.path.exists(prediction_binary_dir):
        os.makedirs(prediction_binary_dir)

    if not os.path.exists(ground_truth_dir):
        os.makedirs(ground_truth_dir)

    transform = transforms.Compose([
        transforms.Resize((256, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.432, 0.433, 0.424), (0.263, 0.264, 0.278)),
    ])

    # Convert Mapillary Labels to Binary masks and store them in ground_truth_dir
    create_ground_truth_binary_mask(mapillary_labels, rgb)

    model = SegmentationModel()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.to(device)
    model.eval()
    test_image_files = [f for f in os.listdir(mapillary_images) if f.endswith('.jpg')]

    with torch.no_grad():
        print('****** Creating Segmentation masks and Binary Masks: ******')
        for image_file in tqdm(test_image_files, position=0, leave=True):
            image_path = os.path.join(mapillary_images, image_file)
            image = Image.open(image_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)

            output = model(image)
            outputx = output.detach().cpu()[0]
            decoded_output = decode_segmap(torch.argmax(outputx, 0))

            numpy_array = (decoded_output * 255).clip(0, 255).astype(np.uint8)
            binary_mask = create_binary_mask(numpy_array, rgb)

            seg_output_image = Image.fromarray(numpy_array)
            bin_output_image = Image.fromarray(binary_mask)

            seg_output_image.save(os.path.join(model_output_folder, f"seg_mask_{image_file}"))
            bin_output_image.save(os.path.join(prediction_binary_dir, f"{image_file.replace('.jpg', '.png')}"), format='PNG')

    print("Segmentation masks saved in:", model_output_folder)
    print("Binary masks saved in:", prediction_binary_dir)
