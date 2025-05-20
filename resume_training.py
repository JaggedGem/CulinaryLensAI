import os
import argparse
import glob
import json
import subprocess
import sys
import torch
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Resume training for Food-101 classifier')
    parser.add_argument('--output-dir', type=str, default='./runs/food101', help='Output directory with checkpoints')
    parser.add_argument('--checkpoint', type=str, default=None, help='Specific checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=None, help='Override number of epochs')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Override device')
    parser.add_argument('--disable-auto-resume', action='store_true', help='Disable auto-resume')
    
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

def get_checkpoint_info(checkpoint_path):
    """Extract information from a checkpoint file"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        info = {
            'epoch': checkpoint.get('epoch', 0),
            'args': checkpoint.get('args', {}),
            'best_acc': checkpoint.get('best_acc', 0)
        }
        return info
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {e}")
        return None

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    
    # Check if output directory exists
    if not output_dir.exists():
        print(f"Output directory {output_dir} does not exist.")
        return
    
    # Find checkpoint to resume from
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint(output_dir)
    
    if checkpoint_path is None:
        print(f"No checkpoint found in {output_dir}")
        return
    
    # Get checkpoint information
    info = get_checkpoint_info(checkpoint_path)
    if info is None:
        return
    
    print(f"\n{'='*60}")
    print(f"Resuming training from checkpoint: {checkpoint_path}")
    print(f"Current epoch: {info['epoch'] + 1}")
    print(f"Best accuracy so far: {info['best_acc']:.4f}")
    print(f"{'='*60}\n")
    
    # Get original training arguments
    orig_args = info.get('args', {})
    
    # Build command for improved training
    cmd = [
        sys.executable, 'improved_training.py',
        '--data-dir', orig_args.get('data_dir', './food-101'),
        '--output-dir', args.output_dir
    ]
    
    # Add arguments from the checkpoint or override from command line
    cmd.extend(['--resume', checkpoint_path])
    
    if not args.disable_auto_resume:
        cmd.append('--auto-resume')
    
    # Use command line arguments to override checkpoint settings if provided
    if args.epochs:
        cmd.extend(['--epochs', str(args.epochs)])
    elif 'epochs' in orig_args:
        cmd.extend(['--epochs', str(orig_args['epochs'])])
    
    if args.batch_size:
        cmd.extend(['--batch-size', str(args.batch_size)])
    elif 'batch_size' in orig_args:
        cmd.extend(['--batch-size', str(orig_args['batch_size'])])
    
    if args.lr:
        cmd.extend(['--lr', str(args.lr)])
    elif 'lr' in orig_args:
        cmd.extend(['--lr', str(orig_args['lr'])])
    
    if args.device:
        cmd.extend(['--device', args.device])
    elif 'device' in orig_args:
        cmd.extend(['--device', orig_args['device']])
    
    # Add other arguments from checkpoint
    if 'model' in orig_args:
        cmd.extend(['--model', orig_args['model']])
    
    if 'num_workers' in orig_args:
        cmd.extend(['--num-workers', str(orig_args['num_workers'])])
    
    if orig_args.get('mixed_precision', True):
        cmd.append('--mixed-precision')
    
    if orig_args.get('use_tensorboard', True):
        cmd.append('--use-tensorboard')
    
    if 'weight_decay' in orig_args:
        cmd.extend(['--weight-decay', str(orig_args['weight_decay'])])
    
    if 'dropout' in orig_args:
        cmd.extend(['--dropout', str(orig_args['dropout'])])
    
    if 'scheduler' in orig_args:
        cmd.extend(['--scheduler', orig_args['scheduler']])
    
    # Print and run the command
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("\nTraining completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with error code {e.returncode}")
    except KeyboardInterrupt:
        print("\nTraining interrupted. You can resume training later by running this script again.")

if __name__ == '__main__':
    main()
