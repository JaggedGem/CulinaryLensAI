import os
import argparse
import subprocess
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Food-101 Classification: Complete Pipeline')
    parser.add_argument('--data-dir', type=str, default='./food-101', help='Directory to store the dataset')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--model', type=str, default='resnet50', 
                        choices=['resnet18', 'resnet50', 'densenet121', 'efficientnet_b0'], 
                        help='Model architecture')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='./runs/food101', help='Output directory')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='cuda' if subprocess.run([sys.executable, '-c', 'import torch; print(torch.cuda.is_available())'], capture_output=True, text=True).stdout.strip() == 'True' else 'cpu', help='Device to use for training')
    parser.add_argument('--skip-dataset', action='store_true', help='Skip dataset preparation')
    parser.add_argument('--skip-training', action='store_true', help='Skip model training')
    parser.add_argument('--demo', action='store_true', help='Launch demo app after training')
    parser.add_argument('--demo-port', type=int, default=7860, help='Port for demo app')
    parser.add_argument('--share', action='store_true', help='Create a public link for the demo app')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Check if requirements are satisfied
    try:
        import torch
        import torchvision
        import tqdm
        import matplotlib
        if args.demo:
            import gradio
    except ImportError as e:
        print(f"Missing requirement: {e}")
        install = input("Would you like to install the missing requirements? (y/n): ")
        if install.lower() == 'y':
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'tqdm', 'matplotlib', 'scikit-learn', 'gradio'])
        else:
            print("Please install the missing requirements and try again.")
            sys.exit(1)
    
    # Prepare dataset
    if not args.skip_dataset:
        print("\n" + "="*50)
        print("Step 1: Preparing Food-101 Dataset")
        print("="*50)
        
        cmd = [sys.executable, 'prepare_food101.py', '--data-dir', args.data_dir]
        subprocess.run(cmd, check=True)
      # Train model
    if not args.skip_training:
        print("\n" + "="*50)
        print("Step 2: Training Model")
        print("="*50)
        
        cmd = [
            sys.executable, 'food101_classifier.py',
            '--data-dir', args.data_dir,
            '--batch-size', str(args.batch_size),
            '--epochs', str(args.epochs),
            '--model', args.model,
            '--lr', str(args.lr),
            '--device', args.device,
            '--output-dir', args.output_dir,
            '--num-workers', str(args.num_workers)
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Training failed with error code {e.returncode}")
            print("Error output:")
            print(e.stderr)
            print("Standard output:")
            print(e.stdout)
    
    # Check if the model exists
    model_path = os.path.join(args.output_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(args.output_dir, 'final_model.pth')
        if not os.path.exists(model_path):
            print(f"No model found in {args.output_dir}")
            if not args.skip_training:
                print("Training might have failed.")
            return
    
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
        
        subprocess.run(cmd)

if __name__ == '__main__':
    main()
