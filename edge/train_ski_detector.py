#!/usr/bin/env python3
"""
Train a YOLO model for ski detection using Roboflow-labeled data.

Usage:
    1. Export dataset from Roboflow in YOLOv8 format
    2. Unzip to a folder (e.g., ski-detection-1/)
    3. Run: python train_ski_detector.py ski-detection-1/

The trained model will be saved to runs/detect/ski_detector/weights/best.pt
"""

import argparse
import os
from pathlib import Path


def train(dataset_path: str, epochs: int = 100, imgsz: int = 1280, model_size: str = 'm'):
    """Train YOLO model for ski detection."""
    from ultralytics import YOLO

    dataset_path = Path(dataset_path)

    # Find data.yaml
    data_yaml = dataset_path / 'data.yaml'
    if not data_yaml.exists():
        # Try looking in subdirectories
        for f in dataset_path.rglob('data.yaml'):
            data_yaml = f
            break

    if not data_yaml.exists():
        print(f"Error: data.yaml not found in {dataset_path}")
        print("Make sure you exported from Roboflow in YOLOv8 format")
        return None

    print(f"Dataset: {data_yaml}")
    print(f"Model: yolov8{model_size}.pt")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}")
    print()

    # Load pretrained model
    model = YOLO(f'yolov8{model_size}.pt')

    # Train
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        name='ski_detector',
        patience=20,  # Early stopping
        save=True,
        plots=True,
        verbose=True
    )

    # Find best model
    best_model = Path('runs/detect/ski_detector/weights/best.pt')
    if best_model.exists():
        print(f"\n✓ Training complete!")
        print(f"  Best model: {best_model}")
        print(f"\nTo use in pose analyzer:")
        print(f"  cp {best_model} ski_detector.pt")
        return best_model
    else:
        print("Training completed but best.pt not found")
        return None


def test_model(model_path: str, image_path: str):
    """Test trained model on an image."""
    from ultralytics import YOLO
    import cv2

    model = YOLO(model_path)
    results = model(image_path)

    # Draw results
    for r in results:
        im = r.plot()
        output_path = 'ski_detection_test.jpg'
        cv2.imwrite(output_path, im)
        print(f"Saved test result to {output_path}")
        os.system(f'open {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Train YOLO ski detector')
    parser.add_argument('dataset', nargs='?', help='Path to Roboflow dataset folder')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--imgsz', type=int, default=1280, help='Image size')
    parser.add_argument('--model', default='m', choices=['n', 's', 'm', 'l', 'x'],
                        help='Model size (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--test', metavar='IMAGE', help='Test trained model on image')
    parser.add_argument('--model-path', default='runs/detect/ski_detector/weights/best.pt',
                        help='Path to trained model for testing')
    args = parser.parse_args()

    if args.test:
        test_model(args.model_path, args.test)
    elif args.dataset:
        train(args.dataset, args.epochs, args.imgsz, args.model)
    else:
        print("Ski Detector Training")
        print("=" * 50)
        print()
        print("Steps:")
        print("1. Upload frames from training_data/ski_frames/ to Roboflow")
        print("2. Use SAM3 to label skis in each image (label: 'ski')")
        print("3. Generate dataset version and export as 'YOLOv8'")
        print("4. Download and unzip the dataset")
        print("5. Run: python train_ski_detector.py <dataset_folder>")
        print()
        print("Example:")
        print("  python train_ski_detector.py ~/Downloads/ski-detection-1/")
        print()
        print("After training, copy the model:")
        print("  cp runs/detect/ski_detector/weights/best.pt ski_detector.pt")


if __name__ == "__main__":
    main()
