import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
import json
import matplotlib.pyplot as plt
from food101_classifier import get_model

def parse_args():
    parser = argparse.ArgumentParser(description='Food-101 Classifier Inference')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model-path', type=str, default='./runs/food101/best_model.pth', help='Path to model checkpoint')
    parser.add_argument('--class-names', type=str, default='./runs/food101/class_names.json', help='Path to class names JSON file')
    parser.add_argument('--model-type', type=str, default='resnet50', choices=['resnet18', 'resnet50', 'densenet121', 'efficientnet_b0'], help='Model architecture')
    parser.add_argument('--output', type=str, default=None, help='Path to save output visualization')
    parser.add_argument('--top-k', type=int, default=5, help='Show top K predictions')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    
    return parser.parse_args()

def predict(model, image_path, class_names, device, transform=None):
    """Make a prediction for an image"""
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    return probabilities[0]

def visualize_prediction(image_path, predictions, class_names, top_k=5, output_path=None):
    """Visualize the top k predictions"""
    # Get top k predictions
    top_probs, top_classes = torch.topk(predictions, top_k)
    top_probs = top_probs.cpu().numpy()
    top_classes = top_classes.cpu().numpy()
    
    # Get class names for top predictions
    top_class_names = [class_names[i] for i in top_classes]
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot image
    ax1.imshow(image)
    ax1.set_title('Input Image')
    ax1.axis('off')
    
    # Plot predictions
    y_pos = range(top_k)
    ax2.barh(y_pos, top_probs * 100, align='center')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([name.replace('_', ' ').title() for name in top_class_names])
    ax2.set_xlabel('Probability (%)')
    ax2.set_title('Top {} Predictions'.format(top_k))
    ax2.grid(axis='x')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()
    
    # Print predictions
    print("\nPredictions:")
    for i in range(top_k):
        print(f"{i+1}. {top_class_names[i].replace('_', ' ').title()}: {top_probs[i]*100:.2f}%")
    
    return top_class_names[0], top_probs[0]

def main():
    args = parse_args()
    
    # Check if image exists
    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")
    
    # Load class names
    with open(args.class_names, 'r') as f:
        class_names = json.load(f)
    
    # Set device
    device = torch.device(args.device)
    print(f"Using {device} for inference")
    
    # Load model
    model = get_model(args.model_type, num_classes=len(class_names))
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Make prediction
    predictions = predict(model, args.image, class_names, device)
    
    # Visualize prediction
    top_class, confidence = visualize_prediction(
        args.image, 
        predictions, 
        class_names, 
        top_k=args.top_k, 
        output_path=args.output
    )
    
    return top_class, confidence

if __name__ == '__main__':
    main()
