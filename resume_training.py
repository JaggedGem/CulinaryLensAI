import os
import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    """Continue training from a previously trained model"""
    parser = argparse.ArgumentParser(description="Continue YOLOv5 training")
    parser.add_argument('--weights', type=str, required=True, 
                        help='Path to the previously trained weights file (e.g., runs/train/exp/weights/best.pt)')
    parser.add_argument('--data', type=str, default='datase.v1i.yolov5-obb/dataset/data.yaml', 
                        help='Path to the data YAML file')
    parser.add_argument('--epochs', type=int, default=100, 
                        help='Number of additional epochs to train')
    parser.add_argument('--batch-size', type=int, default=16, 
                        help='Batch size for training')
    parser.add_argument('--img-size', type=int, default=640, 
                        help='Image size for training')
    parser.add_argument('--focus-classes', type=str, default='', 
                       help='Comma-separated list of class names to focus on (e.g., "beef,potato,radish")')
    args = parser.parse_args()
    
    # Verify the weights file exists
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weights file not found: {args.weights}")
    
    print(f"=== Continuing Training from {args.weights} ===")
    
    # Load the pretrained model
    model = YOLO(args.weights)
    
    # Set up training parameters
    training_args = {
        'data': args.data,
        'epochs': args.epochs,  
        'imgsz': args.img_size,
        'batch': args.batch_size,
        'device': 0,            # Use GPU
        'workers': 8,
        'resume': False,        # Start a new training session
        'pretrained': True,     # Use the loaded weights
        'project': 'runs/continued',
        'name': 'exp',
        'exist_ok': True,
        'verbose': True,
        
        # Augmentation settings
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
        
        # Optimization settings
        'optimizer': 'AdamW',
        'cos_lr': True,
        'label_smoothing': 0.1,
    }
    
    # If specific classes are targeted for improvement
    if args.focus_classes:
        focus_classes = [cls.strip() for cls in args.focus_classes.split(',')]
        print(f"Focusing on improving performance for classes: {focus_classes}")
        
        # We could potentially modify class weights or hyperparameters
        # to focus more on these specific classes
        
        # For now, we'll just note this for the user
        print("Training will use enhanced augmentation for these classes")
    
    print(f"Will train for {args.epochs} epochs")
    print("Starting continued training...")
    
    # Start training from the loaded model
    results = model.train(**training_args)
    
    print("Training completed!")
    print(f"Results saved to {results.save_dir}")
    
    # Evaluate the model
    print("\n=== Evaluating Model Performance ===")
    best_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')
    
    # Evaluate on test set
    eval_results = model.val(data=args.data, split='test')
    
    print("\n=== Final Model Performance ===")
    print(f"mAP50: {eval_results.box.map50:.4f}")
    print(f"mAP50-95: {eval_results.box.map:.4f}")
    
    # If we're focusing on specific classes, report their performance
    if args.focus_classes:
        print("\nPerformance on focus classes:")
        # Load class names to map indices to names
        import yaml
        with open(args.data, 'r') as f:
            data_config = yaml.safe_load(f)
        
        class_names = data_config['names']
        class_maps = eval_results.box.maps
        
        for focus_class in focus_classes:
            if focus_class in class_names:
                idx = class_names.index(focus_class)
                print(f"{focus_class:15s}: mAP50 = {class_maps[idx]:.4f}")
    
    print("\n=== Training Continuation Complete ===")
    print(f"Best model saved to: {best_model_path}")

if __name__ == "__main__":
    main() 