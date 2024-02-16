# File to compute IOU
import numpy as np
import os
from sklearn.metrics import jaccard_score, precision_score, f1_score
from skimage.io import imread
import sys

def compute_metrics(prediction_dir, ground_truth_dir):
    prediction_files = set(os.listdir(prediction_dir))
    ground_truth_files = set(os.listdir(ground_truth_dir))

    # Get the common files between predictions and ground truths
    common_files = prediction_files.intersection(ground_truth_files)

    if len(common_files) == 0:
        raise ValueError("No matching files found between prediction masks and ground truth masks.")

    ious = []
    precisions = []
    f1_scores = []

    for common_file in common_files:
        pred_path = os.path.join(prediction_dir, common_file)
        gt_path = os.path.join(ground_truth_dir, common_file)

        prediction_mask = imread(pred_path, as_gray=True)
        ground_truth_mask = imread(gt_path, as_gray=True)

        # Check if both masks have the same shape
        if prediction_mask.shape != ground_truth_mask.shape:
            print(f"Skipping {common_file} due to shape mismatch. Prediction shape: {prediction_mask.shape}, Ground truth shape: {ground_truth_mask.shape}")
            continue

        # Convert masks to binary format (0 or 1)
        prediction_mask = (prediction_mask > 0).astype(np.uint8)
        ground_truth_mask = (ground_truth_mask > 0).astype(np.uint8)

        prediction_mask_flat = prediction_mask.ravel()
        ground_truth_mask_flat = ground_truth_mask.ravel()

        ious.append(jaccard_score(ground_truth_mask_flat, prediction_mask_flat, zero_division=1))
        precisions.append(precision_score(ground_truth_mask_flat, prediction_mask_flat, zero_division=1))
        f1_scores.append(f1_score(ground_truth_mask_flat, prediction_mask_flat, zero_division=1))

    return np.mean(ious), np.mean(precisions), np.mean(f1_scores)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python compute_metrics.py /path/to/base_mode_predictions /path/to/tuned_model_predictions"
              "/path/to/ground_truths")
        sys.exit(1)

    base_prediction_dir = sys.argv[1]
    tuned_prediction_dir = sys.argv[2]
    ground_truth_dir = sys.argv[3]

    with open('model-IOU.txt', 'w') as file:
        original_stdout = sys.stdout
        try:
            sys.stdout = file
            # Base metrics
            average_iou, average_precision, average_f1 = compute_metrics(base_prediction_dir, ground_truth_dir)
            print("Base Model metrics on Cityscapes dataset:")
            print(f"Average IOU: {average_iou:.4f}")
            print(f"Average Precision: {average_precision:.4f}")
            print(f"Average F1-score: {average_f1:.4f}")

            # Print to stdout as well
            sys.stdout = original_stdout
            print("Base Model metrics on Cityscapes dataset:")
            print(f"Average IOU: {average_iou:.4f}")
            print(f"Average Precision: {average_precision:.4f}")
            print(f"Average F1-score: {average_f1:.4f}")

            sys.stdout = file
            # Fine-tuned metrics
            average_iou, average_precision, average_f1 = compute_metrics(tuned_prediction_dir, ground_truth_dir)
            print("\nFine-Tuned Model metrics on Mapillary dataset:")
            print(f"Average IOU: {average_iou:.4f}")
            print(f"Average Precision: {average_precision:.4f}")
            print(f"Average F1-score: {average_f1:.4f}")

            # Print to stdout as well
            sys.stdout = original_stdout
            print("\nFine-Tuned Model metrics on Mapillary dataset:")
            print(f"Average IOU: {average_iou:.4f}")
            print(f"Average Precision: {average_precision:.4f}")
            print(f"Average F1-score: {average_f1:.4f}")

        finally:
            sys.stdout = original_stdout

    print('Logs for both models have been successfully written to eval_models.txt.')
