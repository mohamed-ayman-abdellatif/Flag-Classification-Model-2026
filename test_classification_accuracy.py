import os
import glob
from ultralytics import YOLO

def main():
    base_dir = r"C:\Users\mido\Documents\antigravity\focused-babbage"
    model_path = os.path.join(base_dir, "yolov8s_flag_classification_best.pt")
    val_dir = os.path.join(base_dir, "kaggle_classification_dataset")

    if not os.path.exists(model_path):
        print(f"[ERROR] Classification weights not found at {model_path}")
        return

    if not os.path.exists(os.path.join(val_dir, "val")):
        print(f"[ERROR] Validation dataset folder not found at {os.path.join(val_dir, 'val')}")
        return

    print(f"Loading classification model: {model_path}")
    model = YOLO(model_path)

    print(f"Evaluating classification model on validation split: {os.path.join(val_dir, 'val')}...")
    results = model.val(data=val_dir, split="val", imgsz=128, batch=16, verbose=True)
    
    # YOLO classification validation returns metrics
    top1 = results.top1 * 100
    top5 = results.top5 * 100
    print("\n" + "="*40)
    print(f"Validation Top-1 Accuracy: {top1:.2f}%")
    print(f"Validation Top-5 Accuracy: {top5:.2f}%")
    print("="*40)

if __name__ == '__main__':
    main()
