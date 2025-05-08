import os
import argparse
from pathlib import Path

def main():
    """Main function to run the optimized training pipeline"""
    parser = argparse.ArgumentParser(description="Optimized YOLOv5 training pipeline")
    parser.add_argument('--balance', action='store_true', help='Balance the dataset before training')
    parser.add_argument('--target-count', type=int, default=200, help='Target count for each class after balancing')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--img-size', type=int, default=640, help='Image size for training')
    parser.add_argument('--model', type=str, default='auto', help='Model to use (yolov5s, yolov5m, yolov5l, yolov5x, or auto)')
    args = parser.parse_args()
    
    print("=== Optimized YOLOv5 Training Pipeline ===")
    
    # Step 1: Balance the dataset if requested
    if args.balance:
        print("\n=== Step 1: Balancing Dataset ===")
        from balance_dataset import balance_dataset
        data_yaml = balance_dataset(target_count=args.target_count)
    else:
        data_yaml = 'datase.v1i.yolov5-obb/dataset/data.yaml'
    
    # Step 2: Set up optimized hyperparameters
    print("\n=== Step 2: Creating Optimized Hyperparameters ===")
    from improved_training import create_hyp_file
    hyp_file = create_hyp_file()
    
    # Step 3: Determine the best model to use
    print("\n=== Step 3: Selecting Optimal Model ===")
    if args.model == 'auto':
        # Auto-select model based on available models
        model_options = ['yolov5l.pt', 'yolov5m.pt', 'yolov5su.pt', 'yolov5s.pt']
        for model_option in model_options:
            if os.path.exists(model_option):
                model_path = model_option
                break
        else:
            model_path = 'yolov5m.pt'
    else:
        model_path = f"{args.model}.pt"
    
    print(f"Selected model: {model_path}")
    
    # Step 4: Train the model
    print("\n=== Step 4: Training Model with Optimized Settings ===")
    from ultralytics import YOLO
    
    model = YOLO(model_path)
    
    training_args = {
        'data': data_yaml,
        'hyp': hyp_file,
        'epochs': args.epochs,
        'patience': 50,
        'imgsz': args.img_size,
        'batch': args.batch_size,
        'device': 0,
        'workers': 8,
        'optimizer': 'AdamW',
        'cos_lr': True,
        'label_smoothing': 0.1,
        'project': 'runs/optimized',
        'name': 'exp',
        'exist_ok': True,
        'pretrained': True,
        'rect': False,
        'multi_scale': True,
        'sync_bn': True,
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
        'cls_pw': 1.0,
        'obj_pw': 1.0,
        'fl_gamma': 2.0,
    }
    
    print("Starting optimized training...")
    results = model.train(**training_args)
    
    print("Training completed!")
    print(f"Results saved to {results.save_dir}")
    
    # Step 5: Evaluate the trained model
    print("\n=== Step 5: Evaluating Model Performance ===")
    best_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')
    
    # Load best model for evaluation
    best_model = YOLO(best_model_path)
    
    # Evaluate on test set
    eval_results = best_model.val(data=data_yaml, split='test')
    
    print("\n=== Final Model Performance ===")
    print(f"mAP50: {eval_results.box.map50:.4f}")
    print(f"mAP50-95: {eval_results.box.map:.4f}")
    
    # Print per-class results
    import yaml
    with open(data_yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    class_names = config['names']
    maps = eval_results.box.maps
    print("\nPer-class performance:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name:15s}: mAP50 = {maps[i]:.4f}")
    
    print("\n=== Training Pipeline Complete ===")
    print(f"Best model saved to: {best_model_path}")
    print("Use this model for inference on new data.")

if __name__ == "__main__":
    main() 