# Flag Classification Model (YOLOv8s-cls)

This repository contains the standalone pipeline for generating the synthetic classification dataset, training a `yolov8s-cls` flag classifier model on Kaggle GPU, and validating accuracy locally.

## Features
- **Synthetic Dataset Generator**: [generate_classification_dataset.py](generate_classification_dataset.py) generates 15,390 balanced training and validation crops directly from template flags and background negatives.
- **Kaggle GPU Training**: [run_kaggle_classification_training.py](run_kaggle_classification_training.py) automates zipping the dataset, uploading it to Kaggle Datasets, pushing the training notebook, and monitoring the remote T4 training run to completion.
- **Local Verification**: [test_classification_accuracy.py](test_classification_accuracy.py) runs inference on local validation splits to evaluate model metrics.

## Standalone Accuracy
- **Validation Top-1 Accuracy**: **97.68%** across all 255 classes (254 flag categories + 1 background category).

## Setup & Running
1. Set up your Kaggle API key in your environment (`KAGGLE_USERNAME` and `KAGGLE_API_TOKEN`).
2. Run dataset generation:
   ```bash
   python generate_classification_dataset.py
   ```
3. Run Kaggle training orchestrator:
   ```bash
   python run_kaggle_classification_training.py
   ```
4. Run validation:
   ```bash
   python test_classification_accuracy.py
   ```
