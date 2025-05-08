# Optimized YOLOv5 Food Detection Training

This project contains scripts for training an optimized YOLOv5 model for food ingredient detection with a focus on addressing class imbalance and improving overall performance.

## Files

- `main.py`: Basic training script that converts OBB labels to YOLO format
- `convert_labels.py`: Converts OBB (Oriented Bounding Box) labels to standard YOLO format
- `balance_dataset.py`: Balances the dataset by generating synthetic augmentations for underrepresented classes
- `improved_training.py`: Implements advanced training techniques with optimized hyperparameters
- `train_optimized.py`: Complete pipeline that combines dataset balancing and optimized training

## Initial Results

The initial training showed:
- Good performance on Onion (mAP50 = 0.952) and Eggs (mAP50 = 0.941)
- Moderate performance on Chicken (mAP50 = 0.581)
- Poor performance on Beef (mAP50 = 0.0865), Potato (mAP50 = 0.227), and Radish (mAP50 = 0.0557)

## Improvements Implemented

1. **Data Augmentation**:
   - Targeted synthetic augmentations for underrepresented classes
   - Increased variety through flips, rotations, color jitter, scaling, and noise
   - Balanced dataset to ensure equal representation

2. **Model Architecture**:
   - Automatic selection of larger models (YOLOv5m/YOLOv5l) when available
   - Better feature extraction for small objects (like radish and potato)

3. **Hyperparameter Optimization**:
   - Optimized anchor boxes for better small object detection
   - Tuned learning rates, augmentation parameters, and loss functions
   - Added focal loss (gamma=2.0) to handle class imbalance

4. **Training Techniques**:
   - Added class weight balancing
   - Implemented cosine learning rate scheduling
   - Used advanced optimizer (AdamW)
   - Added label smoothing for better generalization

## Usage

### Basic Training
```bash
python main.py
```

### Dataset Balancing
```bash
python balance_dataset.py
```

### Improved Training
```bash
python improved_training.py
```

### Complete Optimized Pipeline
```bash
python train_optimized.py --balance --target-count 200 --epochs 300
```

### Command Line Arguments
- `--balance`: Enable dataset balancing
- `--target-count`: Target count for each class (default: 200)
- `--epochs`: Number of training epochs (default: 300)
- `--batch-size`: Batch size (default: 16)
- `--img-size`: Image size for training (default: 640)
- `--model`: Model to use (yolov5s, yolov5m, yolov5l, or auto)

## Hardware Requirements

- CUDA-capable GPU with 6GB+ VRAM
- At least 16GB system RAM
- SSD for faster data loading

## Expected Improvements

These optimizations are expected to significantly improve performance:
- Beef: from 0.0865 to 0.40+ mAP50
- Potato: from 0.227 to 0.50+ mAP50
- Radish: from 0.0557 to 0.40+ mAP50

## Future Improvements

- Implement test time augmentation (TTA)
- Explore ensemble methods for further accuracy gains
- Experiment with model pruning for faster inference 