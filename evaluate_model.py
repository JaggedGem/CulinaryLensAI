import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import time
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd
from pathlib import Path
import sys

# Import the model architecture from our training scripts
try:
    from improved_training import get_model
except ImportError:
    try:
        from food101_classifier import get_model
    except ImportError:
        print("Error: Could not import get_model from either improved_training.py or food101_classifier.py")
        print("Make sure these files exist in the current directory.")
        sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Food-101 Classifier')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--model-type', type=str, default='efficientnet_b0', 
                        choices=['resnet18', 'resnet34', 'resnet50', 'densenet121', 'efficientnet_b0', 'efficientnet_b2'],
                        help='Model architecture')
    parser.add_argument('--data-dir', type=str, default='./food-101', help='Path to the Food-101 dataset')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to use for evaluation (cuda or cpu)')
    parser.add_argument('--output-dir', type=str, default=None, 
                        help='Directory to save evaluation results (defaults to model directory)')
    parser.add_argument('--class-names', type=str, default=None,
                        help='Path to class names JSON file (defaults to looking in model directory)')
    parser.add_argument('--top-k', type=int, default=5, help='Report top-K accuracy')
    parser.add_argument('--num-samples', type=int, default=20, 
                        help='Number of sample predictions to visualize')
    parser.add_argument('--confusion-matrix', action='store_true', default=True,
                        help='Generate confusion matrix visualization')
    
    return parser.parse_args()

def get_test_loader(data_dir, batch_size=64, num_workers=4):
    """Create data loader for the test set"""
    # Standard preprocessing for evaluation
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create test dataset
    test_dir = os.path.join(data_dir, 'test')
    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found at {test_dir}")
        print("Make sure the dataset has been prepared correctly.")
        sys.exit(1)
        
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    
    # Create test dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    class_names = test_dataset.classes
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Number of classes: {len(class_names)}")
    
    return test_loader, class_names

def evaluate(model, dataloader, device, class_names, top_k=5):
    """Evaluate the model on the test dataset"""
    model.eval()
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    correct = 0
    correct_top_k = 0
    total = 0
    
    batch_times = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader, desc="Evaluating")):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Measure inference time
            start_time = time.time()
            outputs = model(inputs)
            batch_time = time.time() - start_time
            batch_times.append(batch_time)
            
            # Get predictions
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            # Calculate top-k accuracy
            _, top_k_preds = torch.topk(outputs, k=top_k, dim=1)
            correct += (preds == targets).sum().item()
            for i, target in enumerate(targets):
                if target in top_k_preds[i]:
                    correct_top_k += 1
            
            total += targets.size(0)
            
            # Store predictions and targets for metrics calculation
            all_labels.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())
    
    # Calculate accuracy
    accuracy = correct / total
    top_k_accuracy = correct_top_k / total
    
    # Calculate average inference time
    avg_inference_time = sum(batch_times) / len(batch_times)
    avg_time_per_image = avg_inference_time / dataloader.batch_size
    
    print(f"Evaluation Results:")
    print(f"Top-1 Accuracy: {accuracy:.4f}")
    print(f"Top-{top_k} Accuracy: {top_k_accuracy:.4f}")
    print(f"Average inference time: {avg_time_per_image*1000:.2f}ms per image")
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    return {
        'accuracy': accuracy,
        'top_k_accuracy': top_k_accuracy,
        'labels': all_labels,
        'predictions': all_preds,
        'probabilities': all_probs,
        'inference_time': avg_time_per_image
    }

def plot_confusion_matrix(labels, predictions, class_names, output_path):
    """Plot and save the confusion matrix"""
    # Create the confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Plot the confusion matrix
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    # Also plot a normalized version
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm_norm, annot=False, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(output_path.replace('.png', '_normalized.png'))
    plt.close()
    
    return cm

def plot_per_class_accuracy(labels, predictions, class_names, output_path):
    """Plot per-class accuracy"""
    # Compute per-class accuracy
    cm = confusion_matrix(labels, predictions)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    # Sort by accuracy for better visualization
    sorted_indices = np.argsort(per_class_acc)
    sorted_acc = per_class_acc[sorted_indices]
    sorted_names = [class_names[i] for i in sorted_indices]
    
    # Plot
    plt.figure(figsize=(15, 10))
    plt.barh(range(len(sorted_names)), sorted_acc, align='center')
    plt.yticks(range(len(sorted_names)), [n.replace('_', ' ').title() for n in sorted_names])
    plt.xlabel('Accuracy')
    plt.title('Per-Class Accuracy')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return per_class_acc

def plot_sample_predictions(dataloader, model, device, class_names, output_path, num_samples=20):
    """Plot sample predictions (both correct and incorrect)"""
    model.eval()
    
    # Get samples
    all_images = []
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            # Add batch to our collection
            all_images.extend(images.cpu())
            all_labels.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
            if len(all_images) >= 200:  # Collect enough samples to choose from
                break
    
    # Make sure we have both correct and incorrect predictions
    correct_indices = [i for i, (t, p) in enumerate(zip(all_labels, all_preds)) if t == p]
    incorrect_indices = [i for i, (t, p) in enumerate(zip(all_labels, all_preds)) if t != p]
    
    # Select a balanced set of samples
    n_correct = min(num_samples // 2, len(correct_indices))
    n_incorrect = min(num_samples - n_correct, len(incorrect_indices))
    
    # Randomly sample
    np.random.seed(42)
    if n_correct > 0:
        correct_samples = np.random.choice(correct_indices, n_correct, replace=False)
    else:
        correct_samples = []
        
    if n_incorrect > 0:
        incorrect_samples = np.random.choice(incorrect_indices, n_incorrect, replace=False)
    else:
        incorrect_samples = []
    
    samples = list(correct_samples) + list(incorrect_samples)
    np.random.shuffle(samples)
    
    # Define denormalization function
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    
    def denormalize(tensor):
        tensor = tensor.clone().detach()
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor.clamp_(0, 1)
    
    # Plot the samples
    rows = int(np.ceil(len(samples) / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(20, 5 * rows))
    axes = axes.flatten()
    
    for i, idx in enumerate(samples):
        if i >= len(axes):
            break
            
        img = denormalize(all_images[idx])
        true_label = all_labels[idx]
        pred_label = all_preds[idx]
        
        axes[i].imshow(img.permute(1, 2, 0))
        title = f"True: {class_names[true_label].replace('_', ' ').title()}\n"
        title += f"Pred: {class_names[pred_label].replace('_', ' ').title()}"
        
        if true_label == pred_label:
            color = 'green'
        else:
            color = 'red'
            
        axes[i].set_title(title, color=color)
        axes[i].axis('off')
    
    for i in range(len(samples), len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_most_confused_classes(labels, predictions, probabilities, class_names, output_path, top_n=10):
    """Plot the most confused class pairs"""
    cm = confusion_matrix(labels, predictions)
    np.fill_diagonal(cm, 0)  # Zero out the diagonal
    
    # Get the pairs with highest confusion
    confused_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j:
                confused_pairs.append((i, j, cm[i, j]))
    
    # Sort by confusion count
    confused_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Plot top confused pairs
    top_pairs = confused_pairs[:top_n]
    
    plt.figure(figsize=(12, 8))
    pair_labels = [f"{class_names[i].replace('_', ' ').title()}\n→\n{class_names[j].replace('_', ' ').title()}" for i, j, _ in top_pairs]
    confusion_counts = [count for _, _, count in top_pairs]
    
    plt.barh(range(len(pair_labels)), confusion_counts)
    plt.yticks(range(len(pair_labels)), pair_labels)
    plt.xlabel('Number of Confusions')
    plt.title('Most Confused Class Pairs')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def format_time(seconds):
    """Format time in readable form"""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)} min {int(seconds)} sec"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)} hours {int(minutes)} min"

def find_class_names_file(model_path):
    """Locate the class_names.json file based on model path"""
    model_dir = os.path.dirname(model_path)
    class_names_path = os.path.join(model_dir, 'class_names.json')
    
    if os.path.exists(class_names_path):
        return class_names_path
    
    # Look in parent directory as well
    parent_dir = os.path.dirname(model_dir)
    class_names_path = os.path.join(parent_dir, 'class_names.json')
    
    if os.path.exists(class_names_path):
        return class_names_path
    
    return None

def main():
    args = parse_args()
    
    # Make sure we can run on the specified device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    print(f"Using {device} for evaluation")
    
    # Determine output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.model_path)
    
    # Create evaluation subdirectory
    eval_dir = os.path.join(args.output_dir, 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    
    # Get class names
    if args.class_names is None:
        args.class_names = find_class_names_file(args.model_path)
        
    if args.class_names and os.path.exists(args.class_names):
        with open(args.class_names, 'r') as f:
            class_names = json.load(f)
        print(f"Loaded class names from {args.class_names}")
    else:
        print("No class names file found, will use labels from the dataset")
        class_names = None
    
    # Load the test dataset
    test_loader, dataset_class_names = get_test_loader(
        args.data_dir, args.batch_size, args.num_workers
    )
    
    # Use dataset class names if no class names file was found
    if class_names is None:
        class_names = dataset_class_names
    
    # Load the model
    try:
        model = get_model(args.model_type, num_classes=len(class_names))
        model = model.to(device)
        
        # Load model weights
        checkpoint = torch.load(args.model_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        print(f"Model loaded from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Start evaluation
    start_time = time.time()
    print(f"Starting evaluation on test set...")
    
    # Run evaluation
    results = evaluate(model, test_loader, device, class_names, args.top_k)
    
    # Compute classification report
    report = classification_report(
        results['labels'], 
        results['predictions'], 
        target_names=[name.replace('_', ' ').title() for name in class_names],
        output_dict=True
    )
    
    # Save report to CSV
    df = pd.DataFrame(report).transpose()
    report_path = os.path.join(eval_dir, 'classification_report.csv')
    df.to_csv(report_path)
    print(f"Classification report saved to {report_path}")
    
    # Plot confusion matrix
    if args.confusion_matrix:
        print("Generating confusion matrix...")
        cm_path = os.path.join(eval_dir, 'confusion_matrix.png')
        cm = plot_confusion_matrix(
            results['labels'], results['predictions'], class_names, cm_path
        )
        print(f"Confusion matrix saved to {cm_path}")
    
    # Plot per-class accuracy
    print("Generating per-class accuracy plot...")
    acc_path = os.path.join(eval_dir, 'per_class_accuracy.png')
    per_class_acc = plot_per_class_accuracy(
        results['labels'], results['predictions'], class_names, acc_path
    )
    print(f"Per-class accuracy plot saved to {acc_path}")
    
    # Plot sample predictions
    print("Generating sample predictions...")
    samples_path = os.path.join(eval_dir, 'sample_predictions.png')
    plot_sample_predictions(
        test_loader, model, device, class_names, samples_path, args.num_samples
    )
    print(f"Sample predictions saved to {samples_path}")
    
    # Plot most confused classes
    print("Generating most confused classes plot...")
    confused_path = os.path.join(eval_dir, 'most_confused_classes.png')
    plot_most_confused_classes(
        results['labels'], results['predictions'], results['probabilities'],
        class_names, confused_path
    )
    print(f"Most confused classes plot saved to {confused_path}")
    
    # Save overall results
    metrics = {
        'top1_accuracy': results['accuracy'],
        'top_k_accuracy': results['top_k_accuracy'],
        'inference_time_ms': results['inference_time'] * 1000,
        'evaluation_time': time.time() - start_time,
        'model': args.model_type,
        'dataset': args.data_dir,
        'num_classes': len(class_names),
        'test_samples': len(results['labels']),
        'device': str(device),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model_path': args.model_path
    }
    
    # Add per-class metrics
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            'accuracy': float(per_class_acc[i]) if i < len(per_class_acc) else 0.0,
            'precision': report[class_name.replace('_', ' ').title()]['precision'],
            'recall': report[class_name.replace('_', ' ').title()]['recall'],
            'f1_score': report[class_name.replace('_', ' ').title()]['f1-score']
        }
    
    metrics['per_class_metrics'] = per_class_metrics
    
    # Save metrics to JSON
    metrics_path = os.path.join(eval_dir, 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save a summary text file
    summary_path = os.path.join(eval_dir, 'evaluation_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Food-101 Evaluation Summary\n")
        f.write(f"=========================\n\n")
        f.write(f"Model: {args.model_type}\n")
        f.write(f"Checkpoint: {args.model_path}\n")
        f.write(f"Evaluated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {device}\n\n")
        
        f.write(f"Accuracy Metrics\n")
        f.write(f"-----------------\n")
        f.write(f"Top-1 Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Top-{args.top_k} Accuracy: {results['top_k_accuracy']:.4f}\n\n")
        
        f.write(f"Performance Metrics\n")
        f.write(f"-------------------\n")
        f.write(f"Inference time: {results['inference_time']*1000:.2f} ms per image\n")
        f.write(f"Total evaluation time: {format_time(metrics['evaluation_time'])}\n\n")
        
        f.write(f"Dataset Info\n")
        f.write(f"------------\n")
        f.write(f"Number of test samples: {len(results['labels'])}\n")
        f.write(f"Number of classes: {len(class_names)}\n\n")
        
        f.write(f"Top 5 Classes by Accuracy\n")
        f.write(f"------------------------\n")
        # Get top 5 classes by accuracy
        top_classes = sorted(
            [(i, per_class_acc[i]) for i in range(len(class_names)) if i < len(per_class_acc)],
            key=lambda x: x[1], reverse=True
        )[:5]
        
        for i, acc in top_classes:
            f.write(f"{class_names[i].replace('_', ' ').title()}: {acc:.4f}\n")
        
        f.write(f"\n\nBottom 5 Classes by Accuracy\n")
        f.write(f"---------------------------\n")
        # Get bottom 5 classes by accuracy
        bottom_classes = sorted(
            [(i, per_class_acc[i]) for i in range(len(class_names)) if i < len(per_class_acc)],
            key=lambda x: x[1]
        )[:5]
        
        for i, acc in bottom_classes:
            f.write(f"{class_names[i].replace('_', ' ').title()}: {acc:.4f}\n")
    
    print(f"\nEvaluation complete! Summary saved to {summary_path}")
    print(f"All evaluation results saved to {eval_dir}")
    print(f"\nTop-1 Accuracy: {results['accuracy']:.4f}")
    print(f"Top-{args.top_k} Accuracy: {results['top_k_accuracy']:.4f}")
    print(f"Average inference time: {results['inference_time']*1000:.2f} ms per image")
    print(f"Total evaluation time: {format_time(metrics['evaluation_time'])}")

if __name__ == "__main__":
    main()
