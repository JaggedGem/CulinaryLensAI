import os
import argparse
import subprocess
import sys
import json
import glob

def parse_args():
    parser = argparse.ArgumentParser(description='Food-101 Classification: Optimized Pipeline')
    parser.add_argument('--data-dir', type=str, default='./food-101', help='Directory to store the dataset')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--model', type=str, default='efficientnet_b0', 
                        choices=['resnet18', 'resnet34', 'resnet50', 'densenet121', 'efficientnet_b0', 'efficientnet_b2'], 
                        help='Model architecture')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='./runs/food101', help='Output directory')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (cuda or cpu)')
    parser.add_argument('--skip-dataset', action='store_true', help='Skip dataset preparation')
    parser.add_argument('--skip-training', action='store_true', help='Skip model training')
    parser.add_argument('--demo', action='store_true', help='Launch demo app after training')
    parser.add_argument('--demo-port', type=int, default=7860, help='Port for demo app')
    parser.add_argument('--share', action='store_true', help='Create a public link for the demo app')
    parser.add_argument('--auto-resume', action='store_true', default=True, help='Automatically resume from the latest checkpoint')
    parser.add_argument('--mixed-precision', action='store_true', default=True, help='Use mixed precision training')
    parser.add_argument('--use-tensorboard', action='store_true', default=True, help='Use TensorBoard for logging')
    
    return parser.parse_args()

def find_latest_checkpoint(output_dir):
    """Find the latest checkpoint in the output directory"""
    checkpoint_path = os.path.join(output_dir, 'latest_checkpoint.pth')
    if os.path.exists(checkpoint_path):
        return checkpoint_path
    
    # If no latest_checkpoint.pth, look for other checkpoints
    checkpoints = glob.glob(os.path.join(output_dir, '*_checkpoint.pth'))
    if checkpoints:
        # Sort by modification time, most recent first
        return sorted(checkpoints, key=os.path.getmtime, reverse=True)[0]
    
    return None

def main():
    args = parse_args()
    
    # First check if required packages are installed
    try:
        import torch
        import torchvision
        import tqdm
        import matplotlib
        import sklearn
        
        # Check CUDA availability after torch is imported
        cuda_available = torch.cuda.is_available()
        # Override device setting if CUDA is not available
        if args.device == 'cuda' and not cuda_available:
            print("CUDA is not available, falling back to CPU")
            args.device = 'cpu'
        elif cuda_available:
            # Display CUDA information
            print(f"CUDA is available: {cuda_available}")
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
    
        # Check additional optional requirements
        if args.use_tensorboard:
            try:
                import tensorboard
            except ImportError:
                print("TensorBoard not found but continuing without it")
                args.use_tensorboard = False
                
        if args.demo:
            try:
                import gradio
            except ImportError:
                print("Gradio not found. The demo will not run.")
                args.demo = False
                
    except ImportError as e:
        print(f"Missing requirement: {e}")
        install = input("Would you like to install the missing requirements? (y/n): ")
        if install.lower() == 'y':
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'tqdm', 'matplotlib', 'scikit-learn', 'tensorboard', 'gradio', 'pandas', 'seaborn'])
            # Try importing again
            print("Please restart the script after installation")
            sys.exit(0)
        else:
            print("Please install the missing requirements and try again.")
            sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare dataset
    if not args.skip_dataset:
        print("\n" + "="*50)
        print("Step 1: Preparing Food-101 Dataset")
        print("="*50)
        
        cmd = [sys.executable, 'prepare_food101.py', '--data-dir', args.data_dir]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Dataset preparation failed with error code {e.returncode}")
            sys.exit(1)
    
    # Check for resume checkpoint
    resume_checkpoint = None
    if args.auto_resume:
        resume_checkpoint = find_latest_checkpoint(args.output_dir)
        if resume_checkpoint:
            print(f"Found checkpoint to resume training: {resume_checkpoint}")
    
    # Train model
    if not args.skip_training:
        print("\n" + "="*50)
        print("Step 2: Training Model with Optimized Settings")
        print("="*50)
        
        cmd = [
            sys.executable, 'improved_training.py',
            '--data-dir', args.data_dir,
            '--batch-size', str(args.batch_size),
            '--epochs', str(args.epochs),
            '--model', args.model,
            '--lr', str(args.lr),
            '--device', args.device,
            '--output-dir', args.output_dir,
            '--num-workers', str(args.num_workers),
        ]
        
        # Add optional arguments
        if args.mixed_precision:
            cmd.append('--mixed-precision')
        
        if args.use_tensorboard:
            cmd.append('--use-tensorboard')
            
        if args.auto_resume and resume_checkpoint:
            cmd.extend(['--resume', resume_checkpoint])
            
        try:
            print("Running training with command:", ' '.join(cmd))
            process = subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Training failed with error code {e.returncode}")
            # Even if training fails, we might have a checkpoint we can use
            print("Checking for available checkpoints...")
            checkpoint_path = find_latest_checkpoint(args.output_dir)
            if checkpoint_path:
                print(f"Found checkpoint at {checkpoint_path}")
            else:
                print("No checkpoint found, exiting.")
                sys.exit(1)
                
        except KeyboardInterrupt:
            print("\nTraining interrupted by user. The model checkpoint will be saved automatically.")
            print("You can resume training by running the script again with --auto-resume")
            # Don't exit, proceed to check if we have a model to use
    
    # Check if the model exists
    model_path = os.path.join(args.output_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(args.output_dir, 'final_model.pth')
        if not os.path.exists(model_path):
            # Look for any checkpoint
            model_path = find_latest_checkpoint(args.output_dir)
            if not model_path:
                print(f"No model found in {args.output_dir}")
                if not args.skip_training:
                    print("Training might have failed or been interrupted.")
                return
    
    # Check if class_names.json exists
    class_names_path = os.path.join(args.output_dir, 'class_names.json')
    if not os.path.exists(class_names_path):
        print("Warning: class_names.json not found. Creating it from the dataset...")
        try:
            import torchvision.datasets as datasets
            image_datasets = datasets.ImageFolder(os.path.join(args.data_dir, 'train'))
            class_names = image_datasets.classes
            with open(class_names_path, 'w') as f:
                json.dump(class_names, f)
            print(f"Created {class_names_path}")
        except Exception as e:
            print(f"Failed to create class_names.json: {e}")
            print("Please ensure the dataset is prepared correctly.")
    
    # Display model information
    print("\n" + "="*50)
    print("Model Information")
    print("="*50)
    
    # Try to load the model to get information
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'args' in checkpoint:
            trained_args = checkpoint['args']
            print(f"Model architecture: {trained_args.get('model', args.model)}")
            print(f"Training epochs: {trained_args.get('epochs', 'unknown')}")
            print(f"Batch size: {trained_args.get('batch_size', 'unknown')}")
            if 'accuracy' in checkpoint:
                print(f"Best validation accuracy: {checkpoint['accuracy']:.4f}")
        else:
            print(f"Model saved at: {model_path}")
    except Exception as e:
        print(f"Could not load model details: {e}")
    
    print("\n" + "="*50)
    print("Step 3: Model Inference")
    print("="*50)
    print(f"Model trained and saved to {model_path}")
    print("You can now use the model to make predictions on new images:")
    print(f"python food101_inference.py --image <path_to_image> --model-path {model_path} --class-names {args.output_dir}/class_names.json --model-type {args.model}")
    
    # Launch demo app if requested
    if args.demo:
        print("\n" + "="*50)
        print("Step 4: Launching Demo App")
        print("="*50)
        
        cmd = [
            sys.executable, 'food101_demo_app.py',
            '--model-path', model_path,
            '--class-names', os.path.join(args.output_dir, 'class_names.json'),
            '--model-type', args.model,
            '--device', args.device,
            '--port', str(args.demo_port)
        ]
        if args.share:
            cmd.append('--share')
        
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            print("Demo app closed by user.")
        except Exception as e:
            print(f"Error running demo app: {e}")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess terminated by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
