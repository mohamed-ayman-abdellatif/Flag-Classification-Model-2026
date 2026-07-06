# Flag Classification Model (YOLOv8s-cls)

This repository contains the standalone pipeline for generating synthetic datasets, training a `yolov8s-cls` flag classifier model on Kaggle T4 GPU, and validating accuracy.

## Features
- **Dynamic Dataset Splitting**: Prepares, splits, and symlinks the public 296-class flag dataset directly in the Kaggle cloud container.
- **Kaggle GPU Training**: [run_kaggle_classification_training.py](run_kaggle_classification_training.py) automates zipping the notebook metadata, pushing the training notebook to Kaggle kernels, and monitoring the remote Nvidia T4 run to completion.
- **Local Verification**: [test_classification_accuracy.py](test_classification_accuracy.py) runs inference on local validation splits to evaluate model metrics.
- **Synthetic Dataset Generator**: [generate_classification_dataset.py](generate_classification_dataset.py) generates 15,390 balanced training and validation crops directly from template flags and background negatives.

## Standalone Accuracy (296 Classes + Background)
- **Validation Top-1 Accuracy**: **98.58%**
- **Validation Top-5 Accuracy**: **99.92%**

---

## Example Predictions & Curves

Here is an example validation batch showing the model's predictions (predicted class name / confidence score):

![YOLOv8 Flag Classification Predictions](docs/val_batch0_pred.jpg)

### Training Curves
![YOLOv8 Training Curves](docs/results.png)

---

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
