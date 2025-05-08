import os
import yaml
import torch
import shutil
from pathlib import Path
from ultralytics import YOLO

def check_requirements():
    """Check and install required packages if not present"""
    try:
        import ultralytics
    except ImportError:
        print("Installing required packages...")
        os.system('pip install ultralytics')
        os.system('pip install pyyaml')
        os.system('pip install torch torchvision')
        print("Required packages installed successfully!")

def load_config():
    """Load the dataset configuration from data.yaml"""
    data_path = 'datase.v1i.yolov5-obb/dataset/data.yaml'
    with open(data_path, 'r') as f:
        return yaml.safe_load(f), data_path

def train_model():
    """Train the YOLOv5-OBB model"""
    # Load configuration
    config, data_path = load_config()
    
    # Check if dataset exists
    dataset_path = Path(config['path'])
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist!")
    
    # Print dataset structure for verification
    print(f"Dataset path: {dataset_path}")
    print(f"Train images path: {dataset_path / config['train']}")
    print(f"Validation images path: {dataset_path / config['val']}")
    
    # Verify that the paths exist
    if not (dataset_path / config['train']).exists():
        raise FileNotFoundError(f"Train images path {dataset_path / config['train']} does not exist!")
    if not (dataset_path / config['val']).exists():
        raise FileNotFoundError(f"Validation images path {dataset_path / config['val']} does not exist!")
    
    # Initialize YOLOv5 model
    # Use YOLOv5su.pt if available, otherwise use YOLOv5s.pt
    model_path = 'yolov5su.pt' if os.path.exists('yolov5su.pt') else 'yolov5s.pt'
    print(f"Using model: {model_path}")
    model = YOLO(model_path)
    
    # Set up training parameters
    training_args = {
        'data': data_path,
        'epochs': 100,
        'imgsz': 640,
        'batch': 16,
        'device': 0,  # Use GPU
        'workers': 8,
        'project': 'runs/train',
        'name': 'exp',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'SGD',
        'verbose': True,
        'seed': 0,
        'deterministic': True
    }
    
    # Start training
    print("Starting training...")
    results = model.train(**training_args)
    
    print("Training completed!")
    print(f"Results saved to {results.save_dir}")

if __name__ == "__main__":
    # Check and install requirements
    check_requirements()
    
    # Create necessary directories
    os.makedirs('runs/train/exp', exist_ok=True)
    
    # Start training
    train_model()

