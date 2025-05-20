import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import time
import json
from pathlib import Path
import torch.cuda.amp as amp  # For mixed precision training
import signal
import sys

# Try importing optional packages
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Weights & Biases not found. Real-time logging to wandb will be disabled.")

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("TensorBoard not found. TensorBoard logging will be disabled.")

# Global flag for graceful termination
terminate_training = False

def signal_handler(sig, frame):
    """Handle Ctrl+C signal for graceful termination"""
    global terminate_training
    print('\nGraceful termination requested. Saving checkpoint at the end of this epoch...')
    terminate_training = True
    # Don't exit immediately, let the training loop handle it

# Register signal handler for Ctrl+C
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model on Food-101 dataset with optimized parameters')
    parser.add_argument('--data-dir', type=str, default='./food-101', help='Path to Food-101 dataset')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--model', type=str, default='efficientnet_b0', 
                        choices=['resnet18', 'resnet34', 'resnet50', 'densenet121', 'efficientnet_b0', 'efficientnet_b2'], 
                        help='Model architecture')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to use for training')
    parser.add_argument('--output-dir', type=str, default='./runs/food101', help='Output directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--auto-resume', action='store_true', help='Automatically resume from the latest checkpoint')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--mixed-precision', action='store_true', default=True, help='Use mixed precision training')
    parser.add_argument('--use-wandb', action='store_true', default=False, help='Use Weights & Biases for logging')
    parser.add_argument('--use-tensorboard', action='store_true', default=True, help='Use TensorBoard for logging')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing factor')
    parser.add_argument('--scheduler', type=str, default='cosine', 
                        choices=['cosine', 'step', 'plateau', 'onecycle'], help='Learning rate scheduler')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--save-interval', type=int, default=1, help='Save checkpoint every N epochs')
    parser.add_argument('--validation-interval', type=int, default=1, help='Run validation every N epochs')
    
    return parser.parse_args()

def get_model(model_name, num_classes=101, dropout_rate=0.2):
    """Create a model with pretrained weights and modified head for Food-101"""
    if model_name == 'resnet18':
        model = models.resnet18(weights='DEFAULT')
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(model.fc.in_features, num_classes)
        )
    elif model_name == 'resnet34':
        model = models.resnet34(weights='DEFAULT')
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(model.fc.in_features, num_classes)
        )
    elif model_name == 'resnet50':
        model = models.resnet50(weights='DEFAULT')
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(model.fc.in_features, num_classes)
        )
    elif model_name == 'densenet121':
        model = models.densenet121(weights='DEFAULT')
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(model.classifier.in_features, num_classes)
        )
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights='DEFAULT')
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )
    elif model_name == 'efficientnet_b2':
        model = models.efficientnet_b2(weights='DEFAULT')
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    return model

def get_data_loaders(data_dir, batch_size=32, num_workers=4):
    """Create data loaders for training and validation with augmentations"""
    # Advanced data augmentation and normalization for training
    # Standard preprocessing for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2)
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Create training and validation datasets
    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train']),
        'test': datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms['test'])
    }
    
    # Pin memory for faster data transfer to GPU
    pin_memory = torch.cuda.is_available()
    
    # Create training and validation dataloaders
    dataloaders = {
        'train': DataLoader(
            image_datasets['train'], 
            batch_size=batch_size,
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None
        ),
        'test': DataLoader(
            image_datasets['test'], 
            batch_size=batch_size * 2,  # Can use larger batch size for validation
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None
        )
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes
    
    print(f"Dataset sizes: {dataset_sizes}")
    print(f"Number of classes: {len(class_names)}")
    
    return dataloaders, dataset_sizes, class_names

def get_scheduler(scheduler_name, optimizer, epochs, steps_per_epoch=None, **kwargs):
    """Get learning rate scheduler"""
    if scheduler_name == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=epochs // 3, gamma=0.1)
    elif scheduler_name == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
    elif scheduler_name == 'onecycle':
        assert steps_per_epoch is not None, "steps_per_epoch is required for OneCycleLR"
        return optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=kwargs.get('lr', 0.001) * 10, 
            steps_per_epoch=steps_per_epoch, 
            epochs=epochs, 
            pct_start=0.2
        )
    else:
        raise ValueError(f"Scheduler {scheduler_name} not supported")

def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None, use_mixed_precision=False):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_corrects = 0
    
    # Progress bar
    pbar = tqdm(dataloader, desc='Training', unit='batch')
    
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass with optional mixed precision
        with torch.cuda.amp.autocast(enabled=use_mixed_precision):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
        
        # Backward + optimize with or without mixed precision
        if use_mixed_precision and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
        # Update progress bar with current batch loss and accuracy
        batch_loss = loss.item()
        batch_acc = torch.sum(preds == labels.data).double() / inputs.size(0)
        pbar.set_postfix({
            'loss': f'{batch_loss:.4f}',
            'acc': f'{batch_acc:.4f}'
        })

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    
    return epoch_loss, epoch_acc.item()

def validate_epoch(model, dataloader, criterion, device, use_mixed_precision=False):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validating', unit='batch')
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward
            with torch.cuda.amp.autocast(enabled=use_mixed_precision):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            # Store predictions and labels for metrics
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            batch_loss = loss.item()
            batch_acc = torch.sum(preds == labels.data).double() / inputs.size(0)
            pbar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'acc': f'{batch_acc:.4f}'
            })

    # Calculate epoch metrics
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    
    # Calculate top-5 accuracy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    return epoch_loss, epoch_acc.item(), all_preds, all_labels

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device, args):
    """Train the model with full features"""
      # Initialize logging tools
    if args.use_wandb and WANDB_AVAILABLE:
        try:
            wandb.init(
                project="food101-classification", 
                config={
                    "model": args.model,
                    "batch_size": args.batch_size,
                    "learning_rate": args.lr,
                    "epochs": args.epochs,
                    "weight_decay": args.weight_decay,
                    "dropout": args.dropout,
                    "label_smoothing": args.label_smoothing,
                    "scheduler": args.scheduler,
                    "mixed_precision": args.mixed_precision
                }
            )
        except Exception as e:
            print(f"Failed to initialize Weights & Biases: {e}")
            args.use_wandb = False
    
    if args.use_tensorboard and TENSORBOARD_AVAILABLE:
        tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard'))
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    global terminate_training
    start_epoch = 0
    best_acc = 0.0
    
    # Initialize mixed precision training
    scaler = amp.GradScaler() if args.mixed_precision and torch.cuda.is_available() else None
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Resume training if checkpoint exists
    if args.auto_resume:
        checkpoint_path = os.path.join(args.output_dir, 'latest_checkpoint.pth')
        if os.path.exists(checkpoint_path):
            args.resume = checkpoint_path
    
    if args.resume and os.path.isfile(args.resume):
        print(f"Loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0.0)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] and scaler:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] and scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'history' in checkpoint:
            history = checkpoint['history']
        print(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
    
    start_time = time.time()
    
    # Display training setup
    print(f"\n{'='*50}")
    print(f"Training setup:")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Mixed precision: {args.mixed_precision}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs} (starting from {start_epoch})")
    print(f"Optimizer: AdamW with weight decay {args.weight_decay}")
    print(f"Scheduler: {args.scheduler}")
    print(f"{'='*50}\n")
    
    # Training loop
    try:
        for epoch in range(start_epoch, args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            print('-' * 50)

            # Training phase
            train_loss, train_acc = train_epoch(
                model, dataloaders['train'], criterion, optimizer, device, 
                scaler=scaler, use_mixed_precision=args.mixed_precision
            )
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # Validation phase
            if (epoch + 1) % args.validation_interval == 0:
                val_loss, val_acc, all_preds, all_labels = validate_epoch(
                    model, dataloaders['test'], criterion, device, 
                    use_mixed_precision=args.mixed_precision
                )
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                
                # Step the scheduler if needed
                if args.scheduler == 'plateau':
                    scheduler.step(val_acc)
                else:
                    scheduler.step()
                
                # Save history
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                # Save best model
                if val_acc > best_acc:
                    best_acc = val_acc
                    print(f"New best accuracy: {best_acc:.4f}")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                        'scaler_state_dict': scaler.state_dict() if scaler else None,
                        'accuracy': best_acc,
                        'args': vars(args)
                    }, os.path.join(args.output_dir, 'best_model.pth'))
            else:
                # If we don't validate, step the scheduler anyway
                scheduler.step()
              # Logging
            if args.use_wandb and WANDB_AVAILABLE:
                try:
                    log_dict = {'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc}
                    if (epoch + 1) % args.validation_interval == 0:
                        log_dict.update({'val_loss': val_loss, 'val_acc': val_acc})
                    wandb.log(log_dict)
                except Exception as e:
                    print(f"Failed to log to Weights & Biases: {e}")
                    args.use_wandb = False
            
            if args.use_tensorboard:
                tb_writer.add_scalar('Loss/train', train_loss, epoch)
                tb_writer.add_scalar('Accuracy/train', train_acc, epoch)
                if (epoch + 1) % args.validation_interval == 0:
                    tb_writer.add_scalar('Loss/val', val_loss, epoch)
                    tb_writer.add_scalar('Accuracy/val', val_acc, epoch)
            
            # Save checkpoint periodically
            if (epoch + 1) % args.save_interval == 0 or terminate_training or (epoch + 1) == args.epochs:
                checkpoint_path = os.path.join(args.output_dir, 'latest_checkpoint.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'scaler_state_dict': scaler.state_dict() if scaler else None,
                    'best_acc': best_acc,
                    'history': history,
                    'args': vars(args)
                }, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
                
                # Plot and save training history
                plot_history(history, args.output_dir)
            
            if terminate_training:
                print("Training terminated by user request.")
                break
    
    except Exception as e:
        print(f"Error during training: {e}")
        # Save checkpoint on error
        checkpoint_path = os.path.join(args.output_dir, 'error_checkpoint.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'best_acc': best_acc,
            'history': history,
            'args': vars(args)        }, checkpoint_path)
        print(f"Error checkpoint saved to {checkpoint_path}")
        raise
    
    finally:
        # Clean up logging
        if args.use_tensorboard and TENSORBOARD_AVAILABLE and 'tb_writer' in locals():
            tb_writer.close()
        if args.use_wandb and WANDB_AVAILABLE:
            try:
                wandb.finish()
            except:
                pass
    
    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    
    return model, history

def plot_history(history, output_dir):
    """Plot training and validation metrics"""
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    if history['val_loss']:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    if history['val_acc']:
        plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()

def evaluate_model(model, dataloader, device, class_names, output_dir='./runs/food101'):
    """Evaluate the model with detailed metrics"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Evaluating'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate accuracy
    accuracy = np.mean(all_preds == all_labels)
    print(f'Test accuracy: {accuracy:.4f}')
    
    # Calculate top-5 accuracy
    top5_acc = 0
    for i, probs in enumerate(all_probs):
        top5_indices = np.argsort(probs)[-5:]
        if all_labels[i] in top5_indices:
            top5_acc += 1
    top5_acc /= len(all_labels)
    print(f'Top-5 accuracy: {top5_acc:.4f}')
    
    # Create confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix (simplified for large number of classes)
    plt.figure(figsize=(12, 10))
    import seaborn as sns
    sns.heatmap(cm, annot=False, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Save classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    import pandas as pd
    df = pd.DataFrame(report).transpose()
    report_path = os.path.join(output_dir, 'classification_report.csv')
    df.to_csv(report_path)
    print(f'Classification report saved to {report_path}')
    
    # Calculate per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    # Plot per-class accuracy
    plt.figure(figsize=(15, 10))
    plt.bar(range(len(class_names)), per_class_acc)
    plt.xticks(range(len(class_names)), class_names, rotation=90)
    plt.title('Per-Class Accuracy')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_accuracy.png'))
    plt.close()
    
    # Save evaluation metrics
    metrics = {
        'accuracy': float(accuracy),
        'top5_accuracy': float(top5_acc),
        'per_class_accuracy': {class_name: float(acc) for class_name, acc in zip(class_names, per_class_acc)}
    }
    
    with open(os.path.join(output_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return accuracy, top5_acc, metrics

def main():
    """Main function to train and evaluate the model"""
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU")
        args.device = 'cpu'
        
    device = torch.device(args.device)
    print(f"Using {device} for training")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get data loaders
    dataloaders, dataset_sizes, class_names = get_data_loaders(
        args.data_dir, 
        args.batch_size,
        args.num_workers
    )
    
    # Save class names
    with open(os.path.join(args.output_dir, 'class_names.json'), 'w') as f:
        json.dump(class_names, f)
    
    # Get model
    model = get_model(args.model, dropout_rate=args.dropout)
    model = model.to(device)
    
    # Define loss function with label smoothing for better generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    # Use AdamW optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    steps_per_epoch = len(dataloaders['train'])
    scheduler = get_scheduler(
        args.scheduler, 
        optimizer, 
        args.epochs,
        steps_per_epoch=steps_per_epoch,
        lr=args.lr
    )
    
    # Train the model
    model, history = train_model(
        model, 
        dataloaders, 
        dataset_sizes,
        criterion, 
        optimizer, 
        scheduler,
        device,
        args=args
    )
    
    # Evaluate the final model
    print("\nEvaluating final model...")
    evaluate_model(model, dataloaders['test'], device, class_names, args.output_dir)
    
    # Save the final model if it hasn't been saved already
    final_model_path = os.path.join(args.output_dir, 'final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': vars(args),
    }, final_model_path)
    print(f"Final model saved to {final_model_path}")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining terminated by user")
        sys.exit(0)
