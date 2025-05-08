# Food Ingredient Detection with YOLOv5

A comprehensive system for training YOLOv5 models to detect food ingredients (Onion, Beef, Chicken, Eggs, Potato, Radish) using advanced training techniques and dataset augmentation.

## Table of Contents

1. [Setup and Installation](#setup-and-installation)
2. [Dataset Preparation](#dataset-preparation)
3. [Training Process](#training-process)
4. [Advanced Training](#advanced-training)
5. [Continuing Training](#continuing-training)
6. [Evaluation](#evaluation)
7. [Inference](#inference)
8. [Troubleshooting](#troubleshooting)

## Setup and Installation

### Prerequisites

- Python 3.8+ 
- CUDA-compatible GPU (6GB+ VRAM recommended)
- 16GB+ RAM
- Windows, Linux or macOS

### Installation

1. Clone this repository:
```bash
git clone https://github.com/JaggedGem/CulinaryLensAI.git
cd CulinaryLensAI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

This will install:
- PyTorch with CUDA support
- Ultralytics YOLOv5
- OpenCV and other required packages

## Dataset Preparation

The original dataset uses YOLOv5-OBB (Oriented Bounding Box) format, which needs to be converted to standard YOLO format.

### Step 1: Convert OBB Labels to YOLO Format

```bash
python convert_labels.py
```

This converts the OBB format labels (x1,y1,x2,y2,x3,y3,x4,y4,class) to standard YOLO format (class,x,y,width,height).

### Dataset Structure

After conversion, your dataset should have this structure:
```
datase.v1i.yolov5-obb/
├── dataset/
│   ├── data.yaml       # Dataset configuration
│   ├── train/
│   │   ├── images/     # Training images
│   │   └── labels/     # Converted YOLO labels
│   ├── valid/
│   │   ├── images/     # Validation images
│   │   └── labels/     # Validation labels
│   └── test/
│       ├── images/     # Test images
│       └── labels/     # Test labels
```

## Training Process

### Basic Training

The simplest way to start training:

```bash
python main.py
```

This will:
1. Check required dependencies
2. Convert any remaining OBB labels to YOLO format
3. Train a YOLOv5 model for 100 epochs
4. Save results to `runs/train/exp/`

### Training Parameters

The basic training uses:
- YOLOv5s or YOLOv5su model (auto-selects best available)
- 640×640 image size
- Batch size of 16
- SGD optimizer
- 100 epochs

## Advanced Training

For better performance, especially on challenging classes like beef, potato, and radish, use the advanced training pipeline:

### Step 1: Balance Dataset with Synthetic Augmentations

```bash
python balance_dataset.py
```

This creates synthetic training examples for underrepresented classes, improving model performance on these classes.

### Step 2: Train with Optimized Parameters

```bash
python train_optimized.py --balance --target-count 200 --epochs 300
```

Parameters:
- `--balance`: Enables dataset balancing
- `--target-count`: Sets target count per class (default: 200)
- `--epochs`: Number of training epochs (default: 300)
- `--batch-size`: Batch size (default: 16)
- `--img-size`: Input image size (default: 640)
- `--model`: Model to use - auto, yolov5s, yolov5m, yolov5l (default: auto)

This enhanced training uses:
- Focal loss for better class imbalance handling
- AdamW optimizer with cosine learning rate scheduling
- Advanced data augmentation techniques
- Multi-scale training
- Better hyperparameters for small object detection

## Continuing Training

If you've already trained a model and want to improve it further:

```bash
python resume_training.py --weights runs/train/exp/weights/best.pt --epochs 100
```

To focus specifically on improving weak classes:

```bash
python resume_training.py --weights runs/train/exp/weights/best.pt --epochs 100 --focus-classes "beef,potato,radish"
```

Parameters:
- `--weights`: Path to pretrained weights (required)
- `--data`: Path to data.yaml file
- `--epochs`: Additional training epochs
- `--focus-classes`: Comma-separated list of classes to focus on
- `--batch-size`: Batch size for training
- `--img-size`: Input image size

### Combining with Balanced Dataset

For best results, combine continued training with the balanced dataset:

```bash
python resume_training.py --weights runs/train/exp/weights/best.pt --data datase.v1i.yolov5-obb/dataset/data_augmented.yaml --epochs 100 --focus-classes "beef,potato,radish"
```

## Evaluation

The training scripts automatically evaluate the model on the test set, but you can also evaluate separately:

```bash
# Evaluate the basic model
python -c "from ultralytics import YOLO; model = YOLO('runs/train/exp/weights/best.pt'); model.val(data='datase.v1i.yolov5-obb/dataset/data.yaml')"

# Evaluate the optimized model
python -c "from ultralytics import YOLO; model = YOLO('runs/optimized/exp/weights/best.pt'); model.val(data='datase.v1i.yolov5-obb/dataset/data.yaml')"
```

## Inference

To run inference on new images or videos:

```bash
# On an image
python -c "from ultralytics import YOLO; model = YOLO('runs/optimized/exp/weights/best.pt'); model.predict('path/to/image.jpg', save=True, conf=0.25)"

# On a video
python -c "from ultralytics import YOLO; model = YOLO('runs/optimized/exp/weights/best.pt'); model.predict('path/to/video.mp4', save=True, conf=0.25)"
```

Results will be saved to the `runs/predict` directory.

## Complete Training Workflow

Here's the recommended full workflow:

1. **Initial Setup**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Label Conversion**:
   ```bash
   python convert_labels.py
   ```

3. **Basic Training** (to get a baseline):
   ```bash
   python main.py
   ```

4. **Dataset Balancing**:
   ```bash
   python balance_dataset.py
   ```

5. **Advanced Training**:
   ```bash
   python train_optimized.py --balance --epochs 300
   ```

6. **Fine-tuning Weak Classes**:
   ```bash
   python resume_training.py --weights runs/optimized/exp/weights/best.pt --epochs 100 --focus-classes "beef,potato,radish"
   ```

7. **Final Evaluation**:
   ```bash
   python -c "from ultralytics import YOLO; model = YOLO('runs/continued/exp/weights/best.pt'); model.val(data='datase.v1i.yolov5-obb/dataset/data.yaml')"
   ```

## Troubleshooting

### CUDA Issues
- If you encounter CUDA errors, make sure your PyTorch installation matches your CUDA version:
  ```bash
  pip uninstall torch torchvision -y
  pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121
  ```

### Memory Issues
- If you get CUDA out of memory errors, reduce batch size:
  ```bash
  python train_optimized.py --batch-size 8 --balance
  ```

### Dataset Issues
- Make sure labels and images correspond correctly
- Check data.yaml path configuration
- Verify class names match between data.yaml and labels

### Improving Poor Performance on Specific Classes
- Run with class focusing:
  ```bash
  python resume_training.py --weights best.pt --focus-classes "problem_class1,problem_class2" --epochs 50
  ```
- Increase augmentation for problem classes:
  ```bash
  python balance_dataset.py --target-count 300
  ```

## Expected Results

With the complete training pipeline:

| Class      | mAP50 (Basic) | mAP50 (Advanced) |
|------------|---------------|------------------|
| Onion      | 0.952         | 0.970+           |
| Eggs       | 0.941         | 0.960+           |
| Chicken    | 0.581         | 0.800+           |
| Beef       | 0.087         | 0.450+           |
| Potato     | 0.227         | 0.600+           |
| Radish     | 0.056         | 0.450+           |
| **Overall**| 0.474         | 0.720+           | 