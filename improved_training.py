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

def create_hyp_file():
    """Create an optimized hyperparameter file for training"""
    hyp = {
        # Augmentation hyperparameters
        'hsv_h': 0.015,  # image HSV-Hue augmentation (fraction)
        'hsv_s': 0.7,    # image HSV-Saturation augmentation (fraction)
        'hsv_v': 0.4,    # image HSV-Value augmentation (fraction)
        'degrees': 0.5,  # image rotation (+/- deg)
        'translate': 0.2,  # image translation (+/- fraction)
        'scale': 0.7,    # image scale (+/- gain)
        'shear': 0.2,    # image shear (+/- deg)
        'perspective': 0.0,  # image perspective (+/- fraction), range 0-0.001
        'flipud': 0.5,   # image flip up-down (probability)
        'fliplr': 0.5,   # image flip left-right (probability)
        'mosaic': 0.8,   # image mosaic (probability)
        'mixup': 0.5,    # image mixup (probability)
        'copy_paste': 0.3,  # segment copy-paste (probability)
        
        # Loss hyperparameters
        'box': 0.05,     # box loss gain
        'cls': 0.5,      # cls loss gain
        'cls_pw': 1.0,   # cls BCELoss positive_weight
        'obj': 1.0,      # obj loss gain (scale with pixels)
        'obj_pw': 1.0,   # obj BCELoss positive_weight
        'iou_t': 0.20,   # IoU training threshold
        'anchor_t': 4.0, # anchor-multiple threshold
        'fl_gamma': 2.0, # focal loss gamma (efficientDet default gamma=1.5)
        
        # Training hyperparameters
        'lr0': 0.01,     # initial learning rate (SGD=1E-2, Adam=1E-3)
        'lrf': 0.01,     # final OneCycleLR learning rate (lr0 * lrf)
        'momentum': 0.937,  # SGD momentum/Adam beta1
        'weight_decay': 0.0005,  # optimizer weight decay 5e-4
        'warmup_epochs': 3.0,  # warmup epochs (fractions ok)
        'warmup_momentum': 0.8,  # warmup initial momentum
        'warmup_bias_lr': 0.1,  # warmup initial bias lr
    }
    
    hyp_path = 'optimized_hyp.yaml'
    with open(hyp_path, 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    
    return hyp_path

def train_model():
    """Train the YOLOv5 model with improved settings"""
    # Load configuration
    config, data_path = load_config()
    
    # Create optimized hyperparameters file
    hyp_path = create_hyp_file()
    
    # Check if dataset exists
    dataset_path = Path(config['path'])
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist!")
    
    # Print dataset structure for verification
    print(f"Dataset path: {dataset_path}")
    print(f"Train images path: {dataset_path / config['train']}")
    print(f"Validation images path: {dataset_path / config['val']}")
    
    # Initialize a larger model (YOLOv5m or YOLOv5l if available)
    model_options = ['yolov5l.pt', 'yolov5m.pt', 'yolov5su.pt', 'yolov5s.pt']
    for model_option in model_options:
        if os.path.exists(model_option):
            model_path = model_option
            break
    else:
        # If no local models found, download YOLOv5m
        model_path = 'yolov5m.pt'
        
    print(f"Using model: {model_path}")
    model = YOLO(model_path)
    
    # Calculate class weights to address imbalance
    class_names = config['names']
    num_classes = len(class_names)
    print(f"Training on {num_classes} classes: {class_names}")
    
    # Set up advanced training parameters
    training_args = {
        'data': data_path,
        'hyp': hyp_path,        # Use optimized hyperparameters
        'epochs': 300,          # Increased epochs
        'patience': 50,         # Early stopping patience
        'imgsz': 640,
        'batch': 16,
        'device': 0,            # Use GPU
        'workers': 8,
        'optimizer': 'AdamW',   # Better optimizer
        'cos_lr': True,         # Cosine learning rate scheduler
        'label_smoothing': 0.1, # Label smoothing
        'project': 'runs/improved_train',
        'name': 'exp',
        'exist_ok': True,
        'pretrained': True,
        'rect': False,          # For more augmentation flexibility
        'multi_scale': True,    # Multi-scale training
        'sync_bn': True,        # Sync batch norm
        'verbose': True,
        'seed': 0,
        'deterministic': True,
        
        # Advanced augmentation
        'degrees': 0.5,
        'translate': 0.2,
        'scale': 0.7,
        'shear': 0.2,
        'perspective': 0.001,
        'flipud': 0.5,
        'fliplr': 0.5,
        'mosaic': 0.8,
        'mixup': 0.5,
        'copy_paste': 0.3,
        
        # Class imbalance handling
        'cls_pw': 1.0,          # Class positive weights
        'obj_pw': 1.0,          # Object positive weights
        'fl_gamma': 2.0,        # Focal loss gamma
    }
    
    # Start training
    print("Starting improved training...")
    results = model.train(**training_args)
    
    print("Training completed!")
    print(f"Results saved to {results.save_dir}")
    
    # Return the best model path
    return os.path.join(results.save_dir, 'weights', 'best.pt')

def evaluate_model(model_path):
    """Evaluate the trained model on the test set"""
    model = YOLO(model_path)
    
    # Load configuration for test set path
    config, _ = load_config()
    dataset_path = Path(config['path'])
    test_path = str(dataset_path / config['test'])
    
    print(f"Evaluating model on test set: {test_path}")
    results = model.val(data=test_path)
    
    print("Evaluation completed!")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    
    # Print per-class results
    class_names = config['names']
    maps = results.box.maps
    print("\nPer-class performance:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name:15s}: mAP50 = {maps[i]:.4f}")

if __name__ == "__main__":
    # Check and install requirements
    check_requirements()
    
    # Create necessary directories
    os.makedirs('runs/improved_train/exp', exist_ok=True)
    
    # Start training with improved parameters
    best_model_path = train_model()
    
    # Evaluate the model
    evaluate_model(best_model_path) 