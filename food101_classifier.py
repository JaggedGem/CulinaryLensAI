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

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model on Food-101 dataset')
    parser.add_argument('--data-dir', type=str, default='./food-101', help='Path to Food-101 dataset')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--model', type=str, default='resnet50', 
                        choices=['resnet18', 'resnet50', 'densenet121', 'efficientnet_b0'], 
                        help='Model architecture')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to use for training')
    parser.add_argument('--output-dir', type=str, default='./runs/food101', help='Output directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for data loading')
    
    return parser.parse_args()

def get_model(model_name, num_classes=101):
    """Create a model with pretrained weights and modified head for Food-101"""
    if model_name == 'resnet18':
        model = models.resnet18(weights='DEFAULT')
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet50':
        model = models.resnet50(weights='DEFAULT')
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'densenet121':
        model = models.densenet121(weights='DEFAULT')
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights='DEFAULT')
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    return model

def get_data_loaders(data_dir, batch_size=32, num_workers=4):
    """Create data loaders for training and validation"""
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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
    
    # Create training and validation dataloaders
    dataloaders = {
        'train': DataLoader(
            image_datasets['train'], 
            batch_size=batch_size,
            shuffle=True, 
            num_workers=num_workers
        ),
        'test': DataLoader(
            image_datasets['test'], 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers
        )
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes
    
    print(f"Dataset sizes: {dataset_sizes}")
    print(f"Number of classes: {len(class_names)}")
    
    return dataloaders, dataset_sizes, class_names

def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=25, output_dir='./runs/food101'):
    """Train the model"""
    os.makedirs(output_dir, exist_ok=True)
    
    start_time = time.time()
    best_acc = 0.0
    
    # Save training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            pbar = tqdm(dataloaders[phase], desc=f'{phase} [{epoch+1}/{num_epochs}]')
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Update progress bar
                batch_loss = loss.item()
                pbar.set_postfix({'loss': f'{batch_loss:.4f}'})

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Update learning rate
            if phase == 'train' and scheduler is not None:
                scheduler.step()
                
            # Save history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            # Save the best model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': best_acc,
                }, os.path.join(output_dir, 'best_model.pth'))
                
        # Save checkpoint after each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_acc': best_acc,
            'history': history,
        }, os.path.join(output_dir, 'latest_checkpoint.pth'))
        
        print()

    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    
    # Plot training history
    plot_history(history, output_dir)
    
    return model, history

def plot_history(history, output_dir):
    """Plot training and validation metrics"""
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()

def evaluate_model(model, dataloader, device, class_names, output_dir='./runs/food101'):
    """Evaluate the model"""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Evaluating'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f'Test accuracy: {accuracy:.4f}')
    
    # Create confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(20, 20))
    import seaborn as sns
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
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
    df.to_csv(os.path.join(output_dir, 'classification_report.csv'))
    
    return accuracy, report

def predict_image(model, image_path, class_names, device, transform=None):
    """Predict a single image"""
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    from PIL import Image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        _, preds = torch.max(outputs, 1)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        
    pred_class = class_names[preds.item()]
    pred_prob = probs[0][preds.item()].item()
    
    # Get top 5 predictions
    top5_probs, top5_indices = torch.topk(probs, 5)
    top5_classes = [class_names[idx.item()] for idx in top5_indices[0]]
    top5_probs = [prob.item() for prob in top5_probs[0]]
    
    return {
        'class': pred_class,
        'probability': pred_prob,
        'top5': list(zip(top5_classes, top5_probs))
    }

def main():
    args = parse_args()
    
    # Set device
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
    
    # Get model
    model = get_model(args.model)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] and scheduler:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"No checkpoint found at '{args.resume}'")
    
    # Train model
    model, history = train_model(
        model, 
        dataloaders, 
        criterion, 
        optimizer, 
        scheduler,
        device,
        num_epochs=args.epochs,
        output_dir=args.output_dir
    )
    
    # Evaluate model
    print("Evaluating model on test set...")
    evaluate_model(model, dataloaders['test'], device, class_names, args.output_dir)
    
    # Save model and class names
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pth'))
    import json
    with open(os.path.join(args.output_dir, 'class_names.json'), 'w') as f:
        json.dump(class_names, f)
    
    print(f"Training and evaluation complete. Results saved to {args.output_dir}")
    
if __name__ == '__main__':
    main()
